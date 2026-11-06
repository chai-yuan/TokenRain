#!/usr/bin/env python3
"""Build TokenRain JSONL documents and packed uint16 token streams."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenrain.config import Config, resolve_path  # noqa: E402
from tokenrain.data import prepare_data  # noqa: E402
from tokenrain.tokenizer import load_tokenizer  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/tokenrain_100m.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="override a config value, e.g. --set train.seq_len=1024")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config, args.overrides)
    if cfg.seq_len > cfg.model.max_position_embeddings:
        raise SystemExit("train.seq_len exceeds model.max_position_embeddings")
    tok = load_tokenizer(cfg)
    print("[tok]", json.dumps(tok.describe(), ensure_ascii=False))
    manifest = prepare_data(cfg, tok)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    files = manifest["files"]
    print(f"[ok] packed files in {resolve_path(cfg.data.packed_dir)}; "
          f"train tokens={files.get('train.bin', {}).get('tokens', 0):,}; "
          f"val tokens={files.get('val.bin', {}).get('tokens', 0):,}")


if __name__ == "__main__":
    main()
