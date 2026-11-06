"""CPU-friendly correctness tests for TokenRain v0.2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenrain.config import Config, DataConfig, ModelConfig, TrainConfig  # noqa: E402
from tokenrain.data import PackedBatchIterator, write_bin  # noqa: E402
from tokenrain.model import (PartialRotaryEmbedding, TokenRainForCausalLM,  # noqa: E402
                             apply_rotary, build_causal_mask,
                             estimate_parameters, token_shift)
from tokenrain.tokenizer import FixedCharTokenizer  # noqa: E402
from tokenrain.train import Trainer, build_optimizer  # noqa: E402
from tokenrain.utils import count_parameters, load_checkpoint  # noqa: E402


def tiny_model_cfg(**kw) -> ModelConfig:
    base = dict(
        vocab_size=128, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=128, max_position_embeddings=64,
        partial_rope_dim=8, token_shift_dim=16)
    base.update(kw)
    return ModelConfig(**base)


def test_config_and_parameter_budget():
    cfg = Config.from_yaml(ROOT / "configs/tokenrain_100m.yaml")
    assert cfg.model.vocab_size == 8192
    assert cfg.model.num_hidden_layers == 14
    assert cfg.model.hidden_size == 1024
    assert cfg.model.partial_rope_dim == 32
    assert cfg.model.token_shift_dim == 256
    assert cfg.sequences_per_step == 2
    assert estimate_parameters(cfg.model) == 103_856_640
    overridden = Config.from_yaml(
        ROOT / "configs/tokenrain_100m.yaml",
        ["train.learning_rate=0.0001"])
    assert overridden.train.learning_rate == pytest.approx(1e-4)


def test_meta_parameter_count():
    cfg = Config.from_yaml(ROOT / "configs/tokenrain_100m.yaml")
    with torch.device("meta"):
        model = TokenRainForCausalLM(cfg.model)
    total, trainable = count_parameters(model)
    assert total == 103_856_640
    assert trainable == total


def test_tied_embeddings_and_no_bias():
    model = TokenRainForCausalLM(tiny_model_cfg())
    assert model.lm_head.weight is model.embed_tokens.weight
    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None


def test_forward_and_backward():
    cfg = tiny_model_cfg()
    model = TokenRainForCausalLM(cfg)
    x = torch.randint(4, cfg.vocab_size, (2, 16))
    out = model(x, labels=x)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_causal_mask_blocks_future_tokens():
    cfg = tiny_model_cfg()
    model = TokenRainForCausalLM(cfg).eval()
    x = torch.randint(4, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        ref = model(x).logits
        changed = x.clone()
        changed[0, 8:] = torch.randint(4, cfg.vocab_size, (4,))
        after = model(changed).logits
    assert torch.allclose(ref[:, :8], after[:, :8], atol=1e-4)


def test_explicit_mask_matches_is_causal():
    cfg = tiny_model_cfg()
    model = TokenRainForCausalLM(cfg).eval()
    x = torch.randint(4, cfg.vocab_size, (1, 12))
    mask = build_causal_mask(12)
    assert mask.shape == (12, 12) and bool(mask[0, 1]) is False
    with torch.no_grad():
        a = model(x).logits
        b = model(x, attention_mask=mask).logits
    assert torch.allclose(a, b, atol=1e-3)


def test_token_shift():
    x = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    out = token_shift(x, dim=4)
    assert torch.equal(out[:, 0, :4], torch.zeros(2, 4))
    assert torch.equal(out[:, 1:, :4], x[:, :-1, :4])
    assert torch.equal(out[:, :, 4:], x[:, :, 4:])


def test_partial_rope():
    cfg = tiny_model_cfg()
    rope = PartialRotaryEmbedding(cfg)
    cos, sin = rope(16, torch.device("cpu"), torch.float32)
    assert cos.shape == sin.shape == (16, cfg.partial_rope_dim)
    q = torch.randn(2, 4, 16, 16)
    k = torch.randn(2, 2, 16, 16)
    q_out, k_out = apply_rotary(q, k, cos, sin, cfg.partial_rope_dim)
    assert q_out.shape == q.shape and k_out.shape == k.shape
    # The non-rotary half is untouched.
    assert torch.equal(q_out[..., cfg.partial_rope_dim:],
                       q[..., cfg.partial_rope_dim:])
    # Position 0 has cos=1, sin=0.
    assert torch.allclose(q_out[:, :, 0, :], q[:, :, 0, :])
    assert not torch.allclose(
        q_out[:, :, 1, :cfg.partial_rope_dim],
        q[:, :, 1, :cfg.partial_rope_dim], atol=1e-5)
    with pytest.raises(ValueError):
        rope(cfg.max_position_embeddings + 1, torch.device("cpu"), torch.float32)


def test_fixed_char_tokenizer_roundtrip():
    tok = FixedCharTokenizer(
        ["人", "工", "智", "能", "𠮷"], ["，", "。"], vocab_size=512)
    assert tok.piece_size == 267
    assert tok.hanzi_base == 260 and tok.symbol_base == 265

    ids = tok.encode("人A，🙂")
    assert ids == [1, 260, 69, 265, 244, 163, 157, 134, 2]
    assert tok.decode(ids) == "人A，🙂"
    assert tok.decode([0, 1, 2, 3]) == ""
    assert tok.decode([0, 1, 2, 3], skip_special_tokens=False) == \
        "<pad><bos><eos><unk>"
    assert tok.decode([5000]) == ""
    assert tok.encode("𠮷", add_special_tokens=False) == [264]
    assert tok.encode("ABC", add_special_tokens=False) == [69, 70, 71]


def test_actual_vocabulary_layout():
    data_dir = ROOT.parent / "data/tokenizer/tokenrain"
    hanzi_file = data_dir / "hanzi.txt"
    symbols_file = data_dir / "symbols.txt"
    if not (hanzi_file.exists() and symbols_file.exists()):
        pytest.skip("run scripts/build_vocab.py first")
    tok = FixedCharTokenizer.from_files(
        hanzi_file, symbols_file, vocab_size=8192)
    assert len(tok.hanzi) == 7848
    assert len(tok.symbols) == 48
    assert tok.hanzi_base == 260
    assert tok.symbol_base == 8108
    assert tok.piece_size == 8156
    assert tok.vocab_size - tok.piece_size == 36

    text = "人工智能正在改变世界。"
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 11
    assert tok.decode(ids) == text
    emoji = "🙂"
    emoji_ids = tok.encode(emoji, add_special_tokens=False)
    assert len(emoji_ids) == 4
    assert tok.decode(emoji_ids) == emoji
    assert tok.encode("ABC", add_special_tokens=False) == [69, 70, 71]


def test_packed_batch_iterator(tmp_path):
    path = tmp_path / "tiny.bin"
    n = write_bin(path, np.arange(256, dtype=np.uint16), 16)
    assert n == 256
    it = PackedBatchIterator(path, 16, 2, torch.device("cpu"))
    x = it.next_batch()
    assert x.shape == (2, 16)
    assert int(x[0, 0]) == 0 and int(x[1, 0]) == 16
    for _ in range(10):
        it.next_batch()
    assert it.epochs >= 1
    assert set(it.state()) == {"offset_tokens", "epochs"}


def test_optimizer_decay_groups():
    model = TokenRainForCausalLM(tiny_model_cfg())
    opt = build_optimizer(model, TrainConfig())
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert all(g["lr"] == pytest.approx(TrainConfig().learning_rate)
               for g in opt.param_groups)
    norm_ids = {id(p) for n, p in model.named_parameters() if "norm" in n}
    nodecay_ids = {id(p) for p in opt.param_groups[1]["params"]}
    assert norm_ids and norm_ids.issubset(nodecay_ids)


def test_legacy_config_with_warmup_ratio_still_loads():
    cfg = Config.from_dict({"train": {"learning_rate": 1e-4,
                                      "warmup_ratio": 0.02}})
    assert cfg.train.learning_rate == pytest.approx(1e-4)


def _tiny_train_config(tmp_path: Path, name: str, **overrides) -> Config:
    packed = tmp_path / "packed"
    packed.mkdir(exist_ok=True)
    write_bin(packed / "train.bin",
              np.random.default_rng(0).integers(4, 8192, 128, dtype=np.uint16), 8)
    data_dir = ROOT.parent / "data/tokenizer/tokenrain"
    train = dict(
        output_dir=str(tmp_path / name),
        total_tokens=16,
        seq_len=8,
        global_batch_tokens=8,
        micro_batch_size=1,
        precision="fp32",
        learning_rate=2e-4,
        log_every_steps=1,
        eval_interval_tokens=10**9,
        save_interval_tokens=10**9,
        eval_batches=1,
        resume="none",
    )
    train.update(overrides)
    return Config(
        version="test",
        model=tiny_model_cfg(vocab_size=8192),
        data=DataConfig(
            raw_dirs=[],
            hanzi_file=str(data_dir / "hanzi.txt"),
            symbols_file=str(data_dir / "symbols.txt"),
            packed_dir=str(packed),
        ),
        train=TrainConfig(**train),
    )


def _read_train_log(cfg: Config):
    path = Path(cfg.train.output_dir) / "train_log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_fixed_lr_and_init_from(tmp_path):
    data_dir = ROOT.parent / "data/tokenizer/tokenrain"
    if not (data_dir / "hanzi.txt").exists():
        pytest.skip("run scripts/build_vocab.py first")

    base_cfg = _tiny_train_config(tmp_path, "base")
    Trainer(base_cfg).train()
    base_log = _read_train_log(base_cfg)
    assert [r["step"] for r in base_log] == [1, 2]
    assert all(r["lr"] == pytest.approx(2e-4) for r in base_log)

    base_ckpt = Path(base_cfg.train.output_dir) / "latest.pt"
    base_weights = load_checkpoint(base_ckpt)["model"]

    ft_cfg = _tiny_train_config(
        tmp_path, "finetune", total_tokens=8, learning_rate=5e-5,
        init_from=str(base_ckpt))
    trainer = Trainer(ft_cfg)
    assert torch.equal(trainer.model.embed_tokens.weight.cpu(),
                       base_weights["embed_tokens.weight"])
    trainer.train()
    ft_log = _read_train_log(ft_cfg)
    assert ft_log[0]["step"] == 1
    assert all(r["lr"] == pytest.approx(5e-5) for r in ft_log)
