#!/usr/bin/env python3
"""Launch a single-process TokenRain training run from a YAML config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenrain.config import Config, resolve_path  # noqa: E402
from tokenrain.train import Trainer  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/tokenrain_100m.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="override a config value, e.g. --set train.total_tokens=10000000")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config, args.overrides)
    print(f"[run] config={args.config} version={cfg.version} "
          f"output={resolve_path(cfg.train.output_dir)}")
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
