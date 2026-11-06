"""Data pipeline: raw .txt -> Chinese-filtered JSONL -> packed uint16 stream."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from .config import Config, resolve_path
from .tokenizer import FixedCharTokenizer


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF       # CJK Extension A
        or 0x4E00 <= code <= 0x9FFF    # CJK Unified Ideographs
        or 0xF900 <= code <= 0xFAFF    # Compatibility Ideographs
        or 0x20000 <= code <= 0x2FA1F  # Extensions B-F + supplement
    )


def cjk_ratio(text: str) -> float:
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 0.0
    return sum(1 for ch in visible if is_cjk(ch)) / len(visible)


def iter_documents(cfg: Config, stats: Optional[Dict[str, int]] = None
                   ) -> Iterator[Tuple[str, str]]:
    """Yield (text, source) documents that pass the Chinese-ratio filter."""
    stats = stats if stats is not None else {}
    for raw in cfg.data.raw_dirs:
        directory = resolve_path(raw)
        if not directory.exists():
            print(f"[data] warning: raw dir not found: {directory}")
            continue
        if not directory.is_dir():
            print(f"[data] warning: not a directory, skipped: {directory}")
            continue
        for path in sorted(directory.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="replace")
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            buf: List[str] = []
            size = 0
            for para in paragraphs:
                buf.append(para)
                size += len(para)
                if size < cfg.data.doc_chars:
                    continue
                doc = "\n".join(buf)
                if _accept(doc, cfg, stats):
                    yield doc, path.stem
                buf, size = [], 0
            if buf:
                doc = "\n".join(buf)
                if _accept(doc, cfg, stats):
                    yield doc, path.stem


def _accept(doc: str, cfg: Config, stats: Dict[str, int]) -> bool:
    stats["raw_docs"] = stats.get("raw_docs", 0) + 1
    if len(doc) < cfg.data.min_doc_chars:
        stats["skipped_short"] = stats.get("skipped_short", 0) + 1
        return False
    if cjk_ratio(doc) < cfg.data.min_cjk_ratio:
        stats["skipped_cjk"] = stats.get("skipped_cjk", 0) + 1
        return False
    return True


def build_jsonl(cfg: Config) -> Dict[str, object]:
    out_dir = resolve_path(cfg.data.jsonl_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    per_source: Counter = Counter()
    counts: Counter = Counter()
    stats: Dict[str, int] = {}
    files = {split: (out_dir / f"{split}.jsonl").open("w", encoding="utf-8")
             for split in ("train", "val")}
    try:
        for text, source in iter_documents(cfg, stats):
            digest = hashlib.md5(text.encode("utf-8")).hexdigest()
            if digest in seen:
                counts["duplicate"] += 1
                continue
            seen.add(digest)
            idx = per_source[source]
            per_source[source] += 1
            bucket = int(hashlib.md5(
                f"{source}:{idx}".encode()).hexdigest()[:8], 16) % 10000
            split = "val" if bucket < cfg.data.val_ratio * 10000 else "train"
            record = {"text": text, "source": source, "split": split}
            files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for f in files.values():
            f.close()
    return {
        "documents": dict(counts),
        "jsonl_dir": str(out_dir),
        "filter": stats,
    }


def tokenize_jsonl(path: Path,
                   tokenizer: FixedCharTokenizer) -> List[np.ndarray]:
    chunks: List[np.ndarray] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            ids = tokenizer.encode(json.loads(line)["text"])
            if not ids:
                continue
            arr = np.asarray(ids, dtype=np.int64)
            if int(arr.max()) >= 65536:
                raise ValueError("token id exceeds uint16 range")
            chunks.append(arr.astype(np.uint16))
    return chunks


def build_stream(chunks: List[np.ndarray], seed: int) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.uint16)
    order = list(range(len(chunks)))
    random.Random(seed).shuffle(order)
    return np.concatenate([chunks[i] for i in order])


def write_bin(path: Path, stream: np.ndarray, seq_len: int) -> int:
    aligned = (len(stream) // seq_len) * seq_len
    arr = np.ascontiguousarray(stream[:aligned], dtype=np.uint16)
    arr.tofile(path)
    return aligned


def prepare_data(cfg: Config,
                 tokenizer: FixedCharTokenizer) -> Dict[str, object]:
    jsonl_info = build_jsonl(cfg)
    out_dir = resolve_path(cfg.data.packed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "version": cfg.version,
        "seq_len": cfg.seq_len,
        "tokenizer": tokenizer.describe(),
        "corpus": jsonl_info,
        "files": {},
    }
    jsonl_dir = Path(str(jsonl_info["jsonl_dir"]))
    for i, (split, name) in enumerate((("train", "train.bin"),
                                       ("val", "val.bin"))):
        path = jsonl_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        chunks = tokenize_jsonl(path, tokenizer)
        stream = build_stream(chunks, cfg.train.seed + i)
        written = (len(stream) // cfg.seq_len) * cfg.seq_len
        if written < cfg.seq_len:
            print(f"[data] warning: {name} has only {written} tokens "
                  f"(< seq_len={cfg.seq_len}); skipping it")
            (out_dir / name).unlink(missing_ok=True)
            continue
        write_bin(out_dir / name, stream, seq_len=cfg.seq_len)
        manifest["files"][name] = {
            "tokens": written,
            "documents": len(chunks),
        }
        print(f"[data] {name}: {len(chunks):,} docs -> {written:,} tokens "
              f"({len(stream) - written:,} dropped for alignment)")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class PackedBin:
    def __init__(self, path: str | Path, seq_len: int):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size < seq_len * 2:
            raise ValueError(
                f"{path.name} has only {path.stat().st_size} bytes; "
                f"need at least {seq_len * 2} for seq_len={seq_len}")
        self.path = path
        self.seq_len = seq_len
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)
        if self.num_tokens < seq_len:
            raise ValueError(f"{path.name} has only {self.num_tokens} tokens")

    @property
    def num_sequences(self) -> int:
        return self.num_tokens // self.seq_len


class PackedBatchIterator:
    """Sequential batches over a packed stream, wrapping around at the end."""

    def __init__(self, path: str | Path, seq_len: int, batch_size: int,
                 device: torch.device, offset_tokens: int = 0):
        self.bin = PackedBin(path, seq_len)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.offset = int(offset_tokens) % self.bin.num_tokens
        self.epochs = 0

    def state(self) -> Dict[str, int]:
        return {"offset_tokens": self.offset, "epochs": self.epochs}

    def set_offset(self, offset_tokens: int) -> None:
        self.offset = int(offset_tokens) % self.bin.num_tokens

    def next_batch(self) -> torch.Tensor:
        need = self.batch_size * self.seq_len
        if self.bin.num_tokens == 0:
            raise ValueError("packed stream is empty")
        buf = np.empty(need, dtype=np.int64)
        filled = 0
        while filled < need:
            take = min(need - filled, self.bin.num_tokens - self.offset)
            buf[filled:filled + take] = self.bin.data[
                self.offset:self.offset + take]
            filled += take
            self.offset += take
            if self.offset >= self.bin.num_tokens:
                self.offset = 0
                self.epochs += 1
        x = torch.from_numpy(buf).reshape(
            self.batch_size, self.seq_len).to(self.device)
        return x
