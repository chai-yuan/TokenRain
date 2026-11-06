"""Fixed character tokenizer for TokenRain.

One code point from the 7,848-character table (or the 48-symbol table) becomes
one token. ASCII reuses the 0x00-0x7F byte tokens. Everything else falls back
to UTF-8 bytes, so any valid UTF-8 text is lossless.

ID layout (default files):
    0-3          <pad> <bos> <eos> <unk>
    4-259        256 byte tokens
    260-8107     7,848 Hanzi
    8108-8155    48 symbols
    8156-8191    36 reserved rows
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .config import Config, resolve_path

SPECIAL_TOKENS: Dict[str, int] = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}
PAD_ID = SPECIAL_TOKENS["<pad>"]
BOS_ID = SPECIAL_TOKENS["<bos>"]
EOS_ID = SPECIAL_TOKENS["<eos>"]
UNK_ID = SPECIAL_TOKENS["<unk>"]
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}

BYTE_BASE = 4
BYTE_COUNT = 256
HANZI_BASE = BYTE_BASE + BYTE_COUNT


class FixedCharTokenizer:
    """Static lookup tokenizer: no BPE/Unigram, no learned merges."""

    kind = "fixed-char"

    def __init__(self, hanzi: Sequence[str], symbols: Sequence[str],
                 vocab_size: int = 8192):
        self.hanzi: List[str] = list(hanzi)
        self.symbols: List[str] = list(symbols)
        self.vocab_size = int(vocab_size)
        self.hanzi_base = HANZI_BASE
        self.symbol_base = self.hanzi_base + len(self.hanzi)
        self.piece_size = self.symbol_base + len(self.symbols)

        if not self.hanzi:
            raise ValueError("Hanzi table must not be empty")
        if self.piece_size > self.vocab_size:
            raise ValueError(
                f"tokenizer needs {self.piece_size} rows, "
                f"vocab_size is {self.vocab_size}")

        self._validate_table(self.hanzi, "Hanzi")
        self._validate_table(self.symbols, "symbol")

        self.hanzi_to_id = {
            ch: self.hanzi_base + i for i, ch in enumerate(self.hanzi)}
        self.symbol_to_id = {
            ch: self.symbol_base + i for i, ch in enumerate(self.symbols)}
        overlap = set(self.hanzi_to_id) & set(self.symbol_to_id)
        if overlap:
            raise ValueError(f"Hanzi/symbol tables overlap: {sorted(overlap)}")

        self.special_token_ids = dict(SPECIAL_TOKENS)
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID

    @staticmethod
    def _validate_table(table: Sequence[str], name: str) -> None:
        seen = set()
        for ch in table:
            if len(ch) != 1:
                raise ValueError(f"{name} entry is not one code point: {ch!r}")
            if ord(ch) < 0x80:
                raise ValueError(f"ASCII char in {name} table: {ch!r}")
            if ch in seen:
                raise ValueError(f"duplicate {name} entry: {ch!r}")
            seen.add(ch)

    @classmethod
    def from_files(cls, hanzi_file: str | Path, symbols_file: str | Path,
                   vocab_size: int = 8192) -> "FixedCharTokenizer":
        hanzi = _read_char_lines(hanzi_file, "Hanzi")
        symbols = _read_char_lines(symbols_file, "symbol")
        return cls(hanzi, symbols, vocab_size=vocab_size)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = self._encode_body(text)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def _encode_body(self, text: str) -> List[int]:
        ids: List[int] = []
        hanzi = self.hanzi_to_id
        symbols = self.symbol_to_id
        for ch in text:
            tid = hanzi.get(ch)
            if tid is not None:
                ids.append(tid)
                continue
            tid = symbols.get(ch)
            if tid is not None:
                ids.append(tid)
                continue
            code = ord(ch)
            if code < 0x80:
                ids.append(BYTE_BASE + code)
            else:
                ids.extend(BYTE_BASE + b for b in ch.encode("utf-8"))
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        out: List[str] = []
        buf = bytearray()

        def flush_bytes() -> None:
            if buf:
                out.append(bytes(buf).decode("utf-8", errors="replace"))
                buf.clear()

        for tid in ids:
            tid = int(tid)
            if tid < 0 or tid >= self.piece_size:
                continue  # reserved / out-of-range rows never carry text
            if tid < BYTE_BASE:
                if not skip_special_tokens:
                    out.append(self.id_to_piece(tid))
                continue
            if tid < self.hanzi_base:
                buf.append(tid - BYTE_BASE)
                continue
            flush_bytes()
            out.append(self.id_to_piece(tid))
        flush_bytes()
        return "".join(out)

    def id_to_piece(self, tid: int) -> str:
        tid = int(tid)
        if 0 <= tid < BYTE_BASE:
            return ID_TO_SPECIAL[tid]
        if tid < self.hanzi_base:
            return f"<0x{tid - BYTE_BASE:02X}>"
        if tid < self.symbol_base:
            return self.hanzi[tid - self.hanzi_base]
        if tid < self.piece_size:
            return self.symbols[tid - self.symbol_base]
        return "<reserved>"

    def describe(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "hanzi": len(self.hanzi),
            "symbols": len(self.symbols),
            "bytes": BYTE_COUNT,
            "specials": len(SPECIAL_TOKENS),
            "piece_size": self.piece_size,
            "vocab_size": self.vocab_size,
            "reserved": self.vocab_size - self.piece_size,
            "hanzi_base": self.hanzi_base,
            "symbol_base": self.symbol_base,
            "special_token_ids": dict(SPECIAL_TOKENS),
        }


def _read_char_lines(path: str | Path, name: str) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} table not found: {path}")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    chars: List[str] = []
    for line in lines:
        # Do not strip: U+3000 is a legitimate symbol-table entry.
        ch = line.rstrip("\r\n")
        if not ch:
            continue
        if len(ch) != 1:
            raise ValueError(f"{path}: {name} line is not one code point: {ch!r}")
        chars.append(ch)
    return chars


def load_tokenizer(cfg: Config) -> FixedCharTokenizer:
    tok = FixedCharTokenizer.from_files(
        resolve_path(cfg.data.hanzi_file),
        resolve_path(cfg.data.symbols_file),
        vocab_size=cfg.model.vocab_size,
    )
    expected = (cfg.model.pad_token_id, cfg.model.bos_token_id,
                cfg.model.eos_token_id, cfg.model.unk_token_id)
    actual = (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id)
    if expected != actual:
        raise ValueError(f"special-token IDs mismatch: config={expected}, tokenizer={actual}")
    return tok
