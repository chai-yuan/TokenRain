"""Small shared helpers: seeding, device/AMP, logging, checkpoints."""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def amp_settings(precision: str, device: torch.device
                 ) -> Tuple[bool, torch.dtype, Optional[Any]]:
    """CUDA BF16/FP16 autocast; CPU uses FP32 for predictable training."""
    if device.type == "cuda":
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, None
        if precision == "fp16":
            return True, torch.float16, torch.amp.GradScaler("cuda")
        if precision == "bf16":
            print("[amp] BF16 not supported on this GPU; falling back to FP32")
    elif precision != "fp32":
        print(f"[amp] {precision} disabled on CPU; using FP32")
    return False, torch.float32, None


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_logger(log_dir: str | Path, name: str = "tokenrain") -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    shutil.move(str(tmp), str(path))


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def human_count(n: int | float) -> str:
    n = float(n)
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f}{unit}"
    return str(int(n))
