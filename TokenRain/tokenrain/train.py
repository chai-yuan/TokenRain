"""Minimal single-process pretraining loop for TokenRain."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .config import Config, resolve_path
from .data import PackedBatchIterator, PackedBin
from .model import TokenRainForCausalLM, estimate_parameters
from .tokenizer import load_tokenizer
from .utils import (amp_settings, append_jsonl, count_parameters, get_logger,
                    human_count, load_checkpoint, resolve_device,
                    save_checkpoint, set_seed)

BETAS = (0.9, 0.95)
EPS = 1e-8
PROGRESS_INTERVAL_SECONDS = 30.0


def build_optimizer(model: torch.nn.Module,
                    cfg) -> torch.optim.Optimizer:
    """Norm and 1-D parameters get zero weight decay."""
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {"lr": cfg.learning_rate, "betas": BETAS, "eps": EPS}
    if torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        tcfg = cfg.train
        self.device = resolve_device("auto")
        set_seed(tcfg.seed)
        self.output_dir = resolve_path(tcfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.output_dir)

        self.tokenizer = load_tokenizer(cfg)
        self.logger.info("tokenizer: %s",
                         json.dumps(self.tokenizer.describe(), ensure_ascii=False))
        if cfg.seq_len > cfg.model.max_position_embeddings:
            raise ValueError(
                f"seq_len={cfg.seq_len} exceeds max_position_embeddings="
                f"{cfg.model.max_position_embeddings}")

        self.model = TokenRainForCausalLM(cfg.model).to(self.device)
        total, trainable = count_parameters(self.model)
        spec = estimate_parameters(cfg.model)
        self.logger.info(
            "model params: %s (spec %s), trainable %s, device %s",
            f"{total:,}", f"{spec:,}", f"{trainable:,}", self.device)
        if total != spec:
            raise RuntimeError("model parameter count differs from the design spec")

        self.micro = tcfg.micro_batch_size
        self.accum = max(1, math.ceil(cfg.sequences_per_step / self.micro))
        self.tokens_per_step = self.micro * self.accum * cfg.seq_len
        if self.tokens_per_step != tcfg.global_batch_tokens:
            self.logger.warning(
                "global batch adjusted: %d -> %d tokens/step",
                tcfg.global_batch_tokens, self.tokens_per_step)
        self.total_steps = math.ceil(tcfg.total_tokens / self.tokens_per_step)
        self.logger.info(
            "steps=%d accum=%d micro=%d tokens/step=%s lr=%g target=%s",
            self.total_steps, self.accum, self.micro,
            f"{self.tokens_per_step:,}", tcfg.learning_rate,
            human_count(tcfg.total_tokens))

        packed_dir = resolve_path(cfg.data.packed_dir)
        self.train_iter = PackedBatchIterator(
            packed_dir / "train.bin", cfg.seq_len, self.micro, self.device)
        passes = tcfg.total_tokens / max(1, self.train_iter.bin.num_tokens)
        self.logger.info("training stream: %s tokens, ~%.1f passes over data",
                         human_count(self.train_iter.bin.num_tokens), passes)
        if passes > 3:
            self.logger.warning(
                "target tokens repeat the corpus ~%.0f times; add data first",
                passes)

        self.val_bin: Optional[PackedBin] = None
        val_path = packed_dir / "val.bin"
        if val_path.exists():
            try:
                self.val_bin = PackedBin(val_path, cfg.seq_len)
            except ValueError as exc:
                self.logger.warning(
                    "validation stream unusable (%s); evaluation will be skipped",
                    exc)
        if self.val_bin is None:
            self.logger.warning("no validation stream; evaluation will be skipped")

        self.optimizer = build_optimizer(self.model, tcfg)
        self.amp_enabled, self.amp_dtype, self.scaler = amp_settings(
            tcfg.precision, self.device)

        self.step = 0
        self.tokens_seen = 0
        self._last_saved_tokens = -1
        self._copy_assets()
        self._init_from_checkpoint()
        self._resume()
        self._set_fixed_lr()
        self.next_eval = (
            (self.tokens_seen // tcfg.eval_interval_tokens) + 1
        ) * tcfg.eval_interval_tokens
        self.next_save = (
            (self.tokens_seen // tcfg.save_interval_tokens) + 1
        ) * tcfg.save_interval_tokens

    def _autocast(self):
        return torch.autocast(device_type=self.device.type,
                              dtype=self.amp_dtype,
                              enabled=self.amp_enabled)

    def _copy_assets(self) -> None:
        for key in ("hanzi_file", "symbols_file"):
            src = resolve_path(getattr(self.cfg.data, key))
            shutil.copy2(src, self.output_dir / src.name)
        (self.output_dir / "tokenizer_info.json").write_text(
            json.dumps(self.tokenizer.describe(), ensure_ascii=False, indent=2),
            encoding="utf-8")
        (self.output_dir / "config.json").write_text(
            json.dumps(self.cfg.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")

    def _checkpoint_path(self) -> Path:
        return self.output_dir / "latest.pt"

    def _resume_target(self) -> Optional[Path]:
        tcfg = self.cfg.train
        if tcfg.resume == "auto":
            candidate = self._checkpoint_path()
            return candidate if candidate.exists() else None
        if tcfg.resume and tcfg.resume != "none":
            return resolve_path(tcfg.resume)
        return None

    def _init_from_checkpoint(self) -> None:
        source = self.cfg.train.init_from
        if not source or source == "none":
            return
        if self._resume_target() is not None:
            self.logger.warning(
                "resume checkpoint takes precedence; init_from=%s ignored",
                source)
            return
        path = resolve_path(source)
        if not path.exists():
            raise FileNotFoundError(f"init_from checkpoint not found: {path}")
        state = load_checkpoint(path)
        try:
            self.model.load_state_dict(state["model"])
        except (KeyError, RuntimeError) as exc:
            raise RuntimeError(
                f"cannot load model weights from {path}; "
                "check that model.* matches the checkpoint") from exc
        self.logger.info("initialized model weights from %s", path)

    def _set_fixed_lr(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = self.cfg.train.learning_rate

    def _resume(self) -> None:
        path = self._resume_target()
        if path is None or not path.exists():
            self.logger.info("no resume checkpoint; starting a new run")
            return
        state = load_checkpoint(path, map_location="cpu")
        self.model.load_state_dict(state["model"])
        if state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler is not None and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.step = int(state.get("step", 0))
        self.tokens_seen = int(state.get("tokens_seen", 0))
        self._last_saved_tokens = self.tokens_seen
        data_state = state.get("data_state", {})
        self.train_iter.set_offset(int(data_state.get("offset_tokens", 0)))
        self.train_iter.epochs = int(data_state.get("epochs", 0))
        self.logger.info("resumed %s at step %d (%s tokens)",
                         path.name, self.step, human_count(self.tokens_seen))

    def _save(self) -> None:
        if self.tokens_seen == self._last_saved_tokens:
            return
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "config": self.cfg.to_dict(),
            "data_state": self.train_iter.state(),
        }
        save_checkpoint(state, self._checkpoint_path())
        self._last_saved_tokens = self.tokens_seen
        self.logger.info("checkpoint saved: %s (step %d, %s tokens)",
                         self._checkpoint_path().name, self.step,
                         human_count(self.tokens_seen))

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_bin is None:
            return {}
        was_training = self.model.training
        self.model.eval()
        it = PackedBatchIterator(
            self.val_bin.path, self.cfg.seq_len, self.micro, self.device)
        total, count = 0.0, 0
        for _ in range(self.cfg.train.eval_batches):
            x = it.next_batch()
            with self._autocast():
                loss = self.model(x, labels=x).loss
            total += float(loss)
            count += 1
        if was_training:
            self.model.train()
        mean = total / max(1, count)
        return {"loss": round(mean, 4),
                "ppl": round(math.exp(min(mean, 20.0)), 4)}

    def train(self) -> None:
        tcfg = self.cfg.train
        lr = tcfg.learning_rate
        self.model.train()
        log_start = time.time()
        log_tokens = 0
        progress_time = time.time()
        progress_tokens = 0
        while self.tokens_seen < tcfg.total_tokens:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            micro_tokens = self.micro * self.cfg.seq_len
            for micro in range(self.accum):
                x = self.train_iter.next_batch()
                with self._autocast():
                    out = self.model(x, labels=x)
                    loss = out.loss / self.accum
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                step_loss += float(out.loss.detach())
                progress_tokens += micro_tokens

                micro_now = time.time()
                if micro_now - progress_time >= PROGRESS_INTERVAL_SECONDS:
                    elapsed = max(1e-6, micro_now - progress_time)
                    self.logger.info(
                        "step %d/%d | micro %d/%d | %s/%s tok | loss %.4f | %.0f tok/s",
                        self.step + 1, self.total_steps,
                        micro + 1, self.accum,
                        human_count(self.tokens_seen + (micro + 1) * micro_tokens),
                        human_count(tcfg.total_tokens),
                        step_loss / (micro + 1),
                        progress_tokens / elapsed)
                    progress_time, progress_tokens = micro_now, 0
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), tcfg.grad_clip)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.step += 1
            self.tokens_seen += self.tokens_per_step
            log_tokens += self.tokens_per_step
            mean_loss = step_loss / self.accum

            now = time.time()
            if self.step == 1 or self.step % tcfg.log_every_steps == 0:
                elapsed = max(1e-6, now - log_start)
                record = {
                    "step": self.step,
                    "tokens": self.tokens_seen,
                    "loss": round(mean_loss, 5),
                    "lr": lr,
                    "grad_norm": round(float(grad_norm), 5),
                    "tokens_per_sec": round(log_tokens / elapsed, 1),
                    "tokens_per_step": self.tokens_per_step,
                    "precision": tcfg.precision,
                    "seed": tcfg.seed,
                }
                append_jsonl(self.output_dir / "train_log.jsonl", record)
                self.logger.info(
                    "step %d/%d | %s tok | loss %.4f | lr %.2e | grad %.3f | %.0f tok/s",
                    self.step, self.total_steps, human_count(self.tokens_seen),
                    mean_loss, lr, float(grad_norm), log_tokens / elapsed)
                log_start, log_tokens = now, 0
                progress_time, progress_tokens = now, 0

            if self.tokens_seen >= self.next_eval:
                results = self.evaluate()
                if results:
                    append_jsonl(self.output_dir / "eval_log.jsonl", {
                        "step": self.step,
                        "tokens": self.tokens_seen,
                        **results,
                    })
                    self.logger.info("validation: loss %.4f | ppl %.2f",
                                     results["loss"], results["ppl"])
                self.next_eval += tcfg.eval_interval_tokens

            if self.tokens_seen >= self.next_save:
                self._save()
                self.next_save += tcfg.save_interval_tokens

        self._save()
        self.logger.info("training complete: %d steps, %s tokens",
                         self.step, human_count(self.tokens_seen))
