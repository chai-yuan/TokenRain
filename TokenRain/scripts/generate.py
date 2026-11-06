#!/usr/bin/env python3
"""Generate text from a TokenRain checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from tokenrain.config import Config, resolve_path  # noqa: E402
from tokenrain.model import TokenRainForCausalLM  # noqa: E402
from tokenrain.tokenizer import FixedCharTokenizer  # noqa: E402
from tokenrain.utils import load_checkpoint, resolve_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    state = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = Config.from_dict(state["config"])
    model = TokenRainForCausalLM(cfg.model)
    model.load_state_dict(state["model"])
    device = resolve_device(args.device or "auto")
    model = model.to(device).eval()

    ckpt_dir = Path(args.checkpoint).resolve().parent
    hanzi_file = ckpt_dir / "hanzi.txt"
    symbols_file = ckpt_dir / "symbols.txt"
    if not (hanzi_file.exists() and symbols_file.exists()):
        hanzi_file = resolve_path(cfg.data.hanzi_file)
        symbols_file = resolve_path(cfg.data.symbols_file)
    tok = FixedCharTokenizer.from_files(
        hanzi_file, symbols_file, vocab_size=cfg.model.vocab_size)

    prompts = args.prompt or ["从前有一只小乌龟，", "人工智能正在改变世界，"]
    max_new = args.max_new_tokens or 64
    temperature = 0.0 if args.greedy else (
        0.8 if args.temperature is None else args.temperature)
    top_k = 40 if args.top_k is None else args.top_k

    for prompt in prompts:
        ids = [tok.bos_id] + tok.encode(prompt, add_special_tokens=False)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                x, max_new_tokens=max_new, temperature=temperature,
                top_k=top_k, valid_vocab_size=tok.piece_size)
        print(f"\nPROMPT: {prompt}\nOUTPUT: {tok.decode(out[0].tolist())}")


if __name__ == "__main__":
    main()
