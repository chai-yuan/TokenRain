"""TokenRain-100M-Base: decoder-only Transformer with hardware-friendly blocks.

Each block uses RMSNorm, Token-Shift, GQA with head-wise QK-Norm and Partial
RoPE, followed by a two-matrix ReLU² FFN. No bias, no dropout, tied embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


def token_shift(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Replace the first ``dim`` channels with the previous timestep's values.

    Position 0 has no predecessor and uses zeros, which keeps the operation
    strictly causal and trivial to implement on hardware.
    """
    if dim <= 0:
        return x
    if x.size(1) == 1:
        shifted = torch.zeros_like(x[:, :, :dim])
    else:
        shifted = torch.cat(
            [torch.zeros_like(x[:, :1, :dim]), x[:, :-1, :dim]], dim=1)
    return torch.cat([shifted, x[:, :, dim:]], dim=-1)


def chunked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                          ignore_index: int = -100,
                          chunk_size: int = 512) -> torch.Tensor:
    """Token-mean cross entropy without upcasting the full logits tensor."""
    if logits.size(1) == 0:
        raise ValueError("chunked_cross_entropy needs at least one position")
    vocab = logits.size(-1)
    total = logits.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, logits.size(1), chunk_size):
        target = labels[:, start:start + chunk_size].reshape(-1)
        valid = int((target != ignore_index).sum())
        if valid == 0:
            continue
        part = logits[:, start:start + chunk_size, :].reshape(-1, vocab).float()
        total = total + F.cross_entropy(
            part, target, ignore_index=ignore_index, reduction="sum")
        count += valid
    if count == 0:
        return logits[:, :1, :].float().sum() * 0.0
    return total / count


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


class HeadRMSNorm(nn.Module):
    """Per-head RMSNorm used by QK-Norm: weight shape (heads, 1, head_dim)."""

    def __init__(self, n_heads: int, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_heads, 1, head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class PartialRotaryEmbedding(nn.Module):
    """RoPE applied only to the first ``partial_rope_dim`` channels per head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.dim = cfg.partial_rope_dim
        inv_freq = 1.0 / (
            cfg.rope_theta ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = cfg.max_position_embeddings

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds {self.max_seq_len}")
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor,
                 rotary_dim: int):
    # q/k: [B, H, T, D]; cos/sin: [T, rotary_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return (torch.cat([q_rot, q_pass], dim=-1),
            torch.cat([k_rot, k_pass], dim=-1))


def build_causal_mask(seq_len: int,
                      device: Optional[torch.device] = None) -> torch.Tensor:
    """Bool mask, True = allowed to attend."""
    return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()


class GQAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv
        self.rotary_dim = cfg.partial_rope_dim
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=False)
        self.q_norm = HeadRMSNorm(self.n_heads, self.head_dim, cfg.rms_norm_eps)
        self.k_norm = HeadRMSNorm(self.n_kv, self.head_dim, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary(q, k, cos, sin, self.rotary_dim)

        if self.n_rep > 1:
            k = (k[:, :, None, :, :]
                 .expand(B, self.n_kv, self.n_rep, T, self.head_dim)
                 .reshape(B, self.n_heads, T, self.head_dim))
            v = (v[:, :, None, :, :]
                 .expand(B, self.n_kv, self.n_rep, T, self.head_dim)
                 .reshape(B, self.n_heads, T, self.head_dim))

        is_causal = attn_mask is None
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask[None, None, :, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0,
            is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class ReLU2FFN(nn.Module):
    """Two-matrix FFN: down(ReLU(x W_up)^2). No gate, no bias."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(
            cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.relu(self.up_proj(x)).pow(2))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = GQAttention(cfg)
        self.mlp = ReLU2FFN(cfg)
        self.shift_dim = cfg.token_shift_dim

    def forward(self, x, cos, sin, attn_mask=None):
        h = x + self.self_attn(
            token_shift(self.attn_norm(x), self.shift_dim),
            cos, sin, attn_mask)
        h = h + self.mlp(token_shift(self.ffn_norm(h), self.shift_dim))
        return h


class TokenRainForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList(
            [Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.rope = PartialRotaryEmbedding(cfg)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)
        if self.embed_tokens.weight.device.type != "meta":
            with torch.no_grad():
                self.embed_tokens.weight[cfg.pad_token_id].zero_()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if module.weight.device.type == "meta":
                return
            nn.init.normal_(module.weight, mean=0.0,
                            std=self.cfg.initializer_range)

    def forward(self, input_ids, labels=None,
                attention_mask=None) -> CausalLMOutput:
        B, T = input_ids.shape
        if T > self.cfg.max_position_embeddings:
            raise ValueError(
                f"input length {T} > max_position_embeddings "
                f"{self.cfg.max_position_embeddings}")
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rope(T, input_ids.device, hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, attention_mask)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = chunked_cross_entropy(
                logits[:, :-1, :], labels[:, 1:])
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=0.8,
                 top_k=40, eos_token_id=None, valid_vocab_size=None):
        """Simple autoregressive sampling; recomputes the full context each step."""
        was_training = self.training
        self.eval()
        eos_token_id = (self.cfg.eos_token_id
                        if eos_token_id is None else eos_token_id)
        ids = input_ids.clone()
        for _ in range(max_new_tokens):
            window = ids[:, -self.cfg.max_position_embeddings:]
            logits = self.forward(window).logits[:, -1, :].float()
            if valid_vocab_size is not None and valid_vocab_size < logits.size(-1):
                logits[:, valid_vocab_size:] = float("-inf")
            logits[:, self.cfg.pad_token_id] = float("-inf")
            logits[:, self.cfg.unk_token_id] = float("-inf")
            logits[:, self.cfg.bos_token_id] = float("-inf")
            if temperature and temperature > 0:
                logits = logits / temperature
                if top_k and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    vals, idx = torch.topk(logits, k, dim=-1)
                    filtered = torch.full_like(logits, float("-inf"))
                    filtered.scatter_(1, idx, vals)
                    logits = filtered
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
        if was_training:
            self.train()
        return ids


def estimate_parameters(cfg: ModelConfig) -> int:
    """Analytic parameter count, including QK-Norm and tied embeddings."""
    h = cfg.hidden_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    attn = h * q_dim + 2 * h * kv_dim + q_dim * h
    qk_norm = q_dim + kv_dim
    ffn = 2 * h * cfg.intermediate_size
    per_block = attn + qk_norm + ffn + 2 * h
    embeddings = cfg.vocab_size * h * (
        1 if cfg.tie_word_embeddings else 2)
    return cfg.num_hidden_layers * per_block + embeddings + h
