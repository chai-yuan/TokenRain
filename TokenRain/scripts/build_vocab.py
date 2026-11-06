#!/usr/bin/env python3
"""Build the fixed TokenRain vocabulary files.

Reads the official level-1/2/3 Hanzi lists and a character-frequency table
(Wikipedia-derived by default), keeps all level-1/2 characters plus the top
`--keep-level3` level-3 characters, and writes hanzi.txt / symbols.txt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data/tokenizer/tokenrain"

# 48 unique CJK punctuation / full-width symbols. The last entry is U+3000,
# an ideographic space; it must not be stripped when reading the file.
SYMBOLS = ("，。、；：？！“”‘’（）《》〈〉【】〔〕…—～·￥＃＆％＋－／＝＠｜＼＾＿｀＜＞"
           "「」『』・〇　")


def read_chars(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"missing character list: {path}")
    chars = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        ch = line.rstrip("\r\n")
        if not ch:
            continue
        if len(ch) != 1:
            raise SystemExit(f"{path}: line is not one code point: {ch!r}")
        chars.append(ch)
    return chars


def load_frequencies(path: Path) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    if not path.exists():
        print(f"[vocab] warning: frequency file not found: {path}; "
              "falling back to official level-3 order")
        return frequencies
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or len(parts[0]) != 1:
            continue
        try:
            frequencies[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return frequencies


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level1", type=Path, default=DEFAULT_DIR / "level-1.txt")
    ap.add_argument("--level2", type=Path, default=DEFAULT_DIR / "level-2.txt")
    ap.add_argument("--level3", type=Path, default=DEFAULT_DIR / "level-3.txt")
    ap.add_argument("--freq-file", type=Path,
                    default=DEFAULT_DIR / "frequence.txt")
    ap.add_argument("--keep-level3", type=int, default=1348)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()

    level1 = read_chars(args.level1)
    level2 = read_chars(args.level2)
    level3 = read_chars(args.level3)
    if (len(level1), len(level2), len(level3)) != (3500, 3000, 1605):
        raise SystemExit(
            f"unexpected level sizes: {len(level1)}/{len(level2)}/{len(level3)}")
    if len(set(level1 + level2 + level3)) != 8105:
        raise SystemExit("level tables are not 8,105 unique characters")
    if not 0 <= args.keep_level3 <= len(level3):
        raise SystemExit("--keep-level3 out of range")

    frequencies = load_frequencies(args.freq_file)
    ranked = sorted(
        range(len(level3)),
        key=lambda i: (-frequencies.get(level3[i], 0), i))
    kept_indices = set(ranked[:args.keep_level3])
    selected_level3 = [ch for i, ch in enumerate(level3) if i in kept_indices]
    excluded_level3 = [ch for i, ch in enumerate(level3) if i not in kept_indices]

    hanzi = level1 + level2 + selected_level3
    symbols = list(SYMBOLS)
    if len(symbols) != 48 or len(set(symbols)) != 48:
        raise SystemExit("symbol table must contain 48 unique characters")
    if set(symbols) & set(hanzi):
        raise SystemExit("symbol table overlaps the Hanzi table")
    if any(ord(ch) < 0x80 for ch in symbols):
        raise SystemExit("symbol table must not contain ASCII characters")

    piece_size = 4 + 256 + len(hanzi) + len(symbols)
    vocab_size = 8192
    if piece_size > vocab_size:
        raise SystemExit(f"piece_size={piece_size} exceeds vocab_size={vocab_size}")

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hanzi.txt").write_text(
        "\n".join(hanzi) + "\n", encoding="utf-8")
    (out_dir / "symbols.txt").write_text(
        "\n".join(symbols) + "\n", encoding="utf-8")
    (out_dir / "level3_excluded.txt").write_text(
        "\n".join(excluded_level3) + "\n", encoding="utf-8")

    meta = {
        "version": "v0.2",
        "level1": len(level1),
        "level2": len(level2),
        "level3_total": len(level3),
        "level3_kept": len(selected_level3),
        "level3_excluded": len(excluded_level3),
        "hanzi": len(hanzi),
        "symbols": len(symbols),
        "bytes": 256,
        "specials": 4,
        "piece_size": piece_size,
        "vocab_size": vocab_size,
        "reserved": vocab_size - piece_size,
        "frequency_source": str(args.freq_file),
    }
    (out_dir / "vocab_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[vocab] wrote {out_dir}")


if __name__ == "__main__":
    main()
