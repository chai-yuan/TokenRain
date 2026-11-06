"""Configuration for TokenRain-100M-Base v0.2.

Relative paths are resolved against the project root (the parent of ``TokenRain/``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    hidden_size: int = 1024
    num_hidden_layers: int = 14
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 2048
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    partial_rope_dim: int = 32
    token_shift_dim: int = 256
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    pad_token_id: int = 0
    unk_token_id: int = 3
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            "num_attention_heads must be divisible by num_key_value_heads"
        assert self.num_attention_heads * self.head_dim == self.hidden_size, \
            "num_attention_heads * head_dim must equal hidden_size"
        assert 0 < self.partial_rope_dim <= self.head_dim, \
            "partial_rope_dim must be in (0, head_dim]"
        assert self.partial_rope_dim % 2 == 0, "partial_rope_dim must be even"
        assert 0 <= self.token_shift_dim <= self.hidden_size, \
            "token_shift_dim must be in [0, hidden_size]"
        for name in ("pad_token_id", "unk_token_id", "bos_token_id", "eos_token_id"):
            value = getattr(self, name)
            assert 0 <= value < self.vocab_size, f"{name}={value} outside vocabulary"


@dataclass
class DataConfig:
    raw_dirs: List[str] = field(default_factory=lambda: ["llm_corpus_chunks"])
    hanzi_file: str = "data/tokenizer/tokenrain/hanzi.txt"
    symbols_file: str = "data/tokenizer/tokenrain/symbols.txt"
    packed_dir: str = "data/processed/tokenrain_packed"
    val_ratio: float = 0.02
    doc_chars: int = 3000
    min_doc_chars: int = 100
    min_cjk_ratio: float = 0.5

    def __post_init__(self) -> None:
        assert 0.0 <= self.val_ratio < 1.0, "val_ratio must be in [0, 1)"
        assert self.doc_chars > 0 and self.min_doc_chars > 0
        assert 0.0 <= self.min_cjk_ratio <= 1.0, "min_cjk_ratio must be in [0, 1]"

    @property
    def jsonl_dir(self) -> str:
        return str(Path(self.packed_dir) / "jsonl")


@dataclass
class TrainConfig:
    seed: int = 42
    output_dir: str = "outputs/tokenrain_100m_v0.2"
    total_tokens: int = 3_000_000_000
    seq_len: int = 4096
    global_batch_tokens: int = 524_288
    micro_batch_size: int = 4
    precision: str = "bf16"
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every_steps: int = 10
    eval_interval_tokens: int = 100_000_000
    eval_batches: int = 8
    save_interval_tokens: int = 500_000_000
    # Load only model weights from a checkpoint; optimizer and step stay fresh.
    init_from: str = ""
    resume: str = "auto"

    def __post_init__(self) -> None:
        assert self.total_tokens > 0 and self.seq_len > 0
        assert self.global_batch_tokens % self.seq_len == 0, \
            "global_batch_tokens must be divisible by seq_len"
        assert self.micro_batch_size > 0
        assert self.precision in ("bf16", "fp16", "fp32")
        assert self.learning_rate > 0 and self.grad_clip > 0
        assert self.log_every_steps > 0
        assert self.eval_interval_tokens > 0 and self.eval_batches > 0
        assert self.save_interval_tokens > 0


@dataclass
class Config:
    version: str = "v0.2"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def seq_len(self) -> int:
        return self.train.seq_len

    @property
    def sequences_per_step(self) -> int:
        return self.train.global_batch_tokens // self.train.seq_len

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Optional[List[str]] = None) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"override must be key=value, got: {item}")
            key, value = item.split("=", 1)
            _set_path(raw, key.strip().split("."), yaml.safe_load(value))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        train = dict(raw.get("train", {}))
        train.pop("warmup_ratio", None)  # legacy checkpoints saved the old schedule
        return cls(
            version=str(raw.get("version", "v0.2")),
            model=_build(ModelConfig, raw.get("model", {})),
            data=_build(DataConfig, raw.get("data", {})),
            train=_build(TrainConfig, train),
        )


def _build(target, values: Dict[str, Any]):
    allowed = {f.name for f in dataclasses.fields(target)}
    unknown = [key for key in values if key not in allowed]
    if unknown:
        raise ValueError(f"unknown {target.__name__} keys: {unknown}")
    return target(**values)


def _set_path(data: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur = data
    for key in keys[:-1]:
        if not isinstance(cur.get(key), dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value
