#!/usr/bin/env python3
"""Parse a GR00T text training log into a wandb-style summary + optional plot.

HF Trainer prints loss/lr/grad_norm dicts to stdout every ``logging_steps``
(default 10). With wandb disabled (``--use-wandb`` not passed) the only
record of training progress is the raw text log -- and the tqdm progress
bar emits ``\\r`` so the loss dicts are visually hidden when you ``tail`` it
naively. This tool:

1. Converts ``\\r`` -> ``\\n`` so all the per-logging-step dicts become
   visible.
2. Extracts ``loss``, ``learning_rate``, ``grad_norm`` from each dict and
   maps it to a step number (``i * logging_steps``, no-resume assumption).
3. Parses the latest tqdm line for current step + iter/s + ETA.
4. Prints a colourised text summary (current loss, EMA-smoothed loss, lr,
   grad-norm, time-to-next-checkpoint, time-to-finish).
5. Optionally writes a 3-panel PNG (log-scale loss, lr, grad-norm) next
   to the log for visual inspection.

Use ``--watch SECS`` to refresh in place (poor-man's wandb dashboard).

Examples
--------

    # one-shot snapshot of the latest run
    python scripts/inspect_training_log.py --latest

    # specific log + write PNG next to it
    python scripts/inspect_training_log.py \
        --log data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/train_20260608_152637.log \
        --plot

    # refresh every 30 s, no plot (cheap, fine to leave open in a pane)
    python scripts/inspect_training_log.py --latest --watch 30
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from glob import glob
from pathlib import Path

# Single-quoted dict format that HF Trainer prints, e.g.
#   {'loss': '1.355', 'grad_norm': '0.2988', 'learning_rate': '3.6e-07'}
# Numbers are strings (quoted) in this format.
LOSS_RE = re.compile(
    r"\{'loss':\s*'?(?P<loss>[-+0-9.eE]+)'?,\s*"
    r"'grad_norm':\s*'?(?P<grad>[-+0-9.eE]+)'?,\s*"
    r"'learning_rate':\s*'?(?P<lr>[-+0-9.eE]+)'?"
)

# tqdm progress bar line, e.g.
#   "  7%|▋         | 3544/50000 [44:23<10:44:07,  1.20it/s]"
TQDM_RE = re.compile(
    r"(?P<step>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[^<]+)<(?P<eta>[^,]+),\s*"
    r"(?P<rate>[\d.]+)\s*(?:it/s|s/it)"
)

# Match either "1.20it/s" or "1.30s/it"
RATE_RE = re.compile(r"([\d.]+)\s*(it/s|s/it)")


def find_latest_log() -> Path:
    """Locate the most-recent ``train_*.log`` under ``data/checkpoints/``."""
    candidates = sorted(
        glob("data/checkpoints/*/train_*.log"),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            "no train_*.log found under data/checkpoints/. Pass --log explicitly."
        )
    return Path(candidates[0])


def parse_log(log_path: Path, logging_steps: int = 10) -> dict:
    """Read the whole log; return parsed records + latest tqdm snapshot."""
    raw = log_path.read_bytes().decode("utf-8", errors="replace")
    text = raw.replace("\r", "\n")

    records: list[dict] = []
    for i, m in enumerate(LOSS_RE.finditer(text)):
        records.append(
            {
                "step": (i + 1) * logging_steps,
                "loss": float(m["loss"]),
                "grad_norm": float(m["grad"]),
                "lr": float(m["lr"]),
            }
        )

    # Last tqdm progress line is the freshest step/ETA pair.
    last_tqdm = None
    for m in TQDM_RE.finditer(text):
        last_tqdm = m
    tqdm_info: dict = {}
    if last_tqdm is not None:
        rate_val = float(last_tqdm["rate"])
        rate_unit_match = RATE_RE.search(text[last_tqdm.end() - 12 : last_tqdm.end()])
        unit = rate_unit_match.group(2) if rate_unit_match else "it/s"
        # Normalize to it/s.
        it_per_s = rate_val if unit == "it/s" else (1.0 / rate_val if rate_val > 0 else 0.0)
        tqdm_info = {
            "step": int(last_tqdm["step"]),
            "total": int(last_tqdm["total"]),
            "elapsed": last_tqdm["elapsed"].strip(),
            "eta": last_tqdm["eta"].strip(),
            "it_per_s": it_per_s,
        }

    return {"records": records, "tqdm": tqdm_info, "path": log_path}


def ema(values: list[float], alpha: float = 0.05) -> list[float]:
    """Standard exponential moving average (wandb-style smoothing)."""
    out: list[float] = []
    s = None
    for v in values:
        s = v if s is None else alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def fmt_eta_to_checkpoint(current_step: int, save_steps: int, it_per_s: float) -> str:
    if it_per_s <= 0 or save_steps <= 0:
        return "?"
    next_ckpt = ((current_step // save_steps) + 1) * save_steps
    steps_left = next_ckpt - current_step
    secs = steps_left / it_per_s
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"step {next_ckpt} in {h:d}h {m:02d}m {s:02d}s"


def print_summary(parsed: dict, save_steps: int) -> None:
    records = parsed["records"]
    tqdm_info = parsed["tqdm"]
    log_path = parsed["path"]

    if not records:
        print(f"[inspect] no loss records in {log_path}; trainer may still be loading model")
        return

    losses = [r["loss"] for r in records]
    grads = [r["grad_norm"] for r in records]
    lrs = [r["lr"] for r in records]
    smoothed = ema(losses)

    current = records[-1]
    smoothed_now = smoothed[-1]
    first = records[0]
    pct_change = (current["loss"] - first["loss"]) / first["loss"] * 100.0
    pct_drop = -pct_change

    print()
    print(f"=== {log_path} ===")
    if tqdm_info:
        print(
            f"  step:        {tqdm_info['step']:>6d} / {tqdm_info['total']:<6d}"
            f"  ({100 * tqdm_info['step'] / tqdm_info['total']:5.2f}%)"
            f"   {tqdm_info['it_per_s']:.2f} it/s"
        )
        print(f"  elapsed:     {tqdm_info['elapsed']}    remaining (tqdm): {tqdm_info['eta']}")
        if save_steps > 0:
            print(
                f"  next ckpt:   {fmt_eta_to_checkpoint(tqdm_info['step'], save_steps, tqdm_info['it_per_s'])}"
            )
    print()
    print(f"  loss  raw  : {current['loss']:.4g}      smoothed (ema): {smoothed_now:.4g}")
    direction = "drop" if pct_drop >= 0 else "rise"
    print(
        f"  loss  {direction:<5}: {first['loss']:.4g} -> {current['loss']:.4g}   "
        f"({abs(pct_drop):.1f}% {direction} vs first record)"
    )
    print(f"  lr        : {current['lr']:.3g}        grad_norm: {current['grad_norm']:.3g}")
    print(f"  records   : {len(records)} (every {records[1]['step'] - records[0]['step'] if len(records) > 1 else '?'} steps)")
    print()


def write_plot(parsed: dict, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[inspect] matplotlib not available; skipping --plot", file=sys.stderr)
        return

    records = parsed["records"]
    if not records:
        print("[inspect] no records to plot", file=sys.stderr)
        return

    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    grads = [r["grad_norm"] for r in records]
    lrs = [r["lr"] for r in records]
    smoothed = ema(losses)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax = axes[0]
    ax.plot(steps, losses, alpha=0.3, color="C0", label="raw")
    ax.plot(steps, smoothed, color="C0", linewidth=2, label="ema(α=0.05)")
    ax.set_yscale("log")
    ax.set_ylabel("loss (log)")
    ax.set_title(parsed["path"].name)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")

    axes[1].plot(steps, lrs, color="C2")
    axes[1].set_ylabel("learning rate")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, grads, color="C3", alpha=0.5)
    axes[2].plot(steps, ema(grads, alpha=0.05), color="C3", linewidth=2)
    axes[2].set_ylabel("grad_norm")
    axes[2].set_xlabel("step")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[inspect] wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", type=Path, help="path to a train_*.log file")
    p.add_argument(
        "--latest",
        action="store_true",
        help="auto-pick the newest train_*.log under data/checkpoints/",
    )
    p.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="value HF Trainer was configured with (default 10)",
    )
    p.add_argument(
        "--save-steps",
        type=int,
        default=10_000,
        help="save_steps to compute next-checkpoint ETA (default 10000)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="write loss/lr/grad PNG next to the log",
    )
    p.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="re-print every N seconds (Ctrl-C to stop)",
    )
    args = p.parse_args()

    if args.log is None and not args.latest:
        p.error("pass --log PATH or --latest")
    log_path = args.log if args.log else find_latest_log()
    if not log_path.exists():
        raise SystemExit(f"log not found: {log_path}")

    def one_pass() -> None:
        parsed = parse_log(log_path, logging_steps=args.logging_steps)
        print_summary(parsed, save_steps=args.save_steps)
        if args.plot:
            # Embed the latest step into the filename so external image viewers
            # (e.g. Cursor's chat) don't show a cached older render. Also keep a
            # stable "latest" symlink/copy at <log>.png for convenience.
            latest_step = parsed["tqdm"].get("step") if parsed["tqdm"] else None
            if latest_step is None and parsed["records"]:
                latest_step = parsed["records"][-1]["step"]
            stem = log_path.stem
            stamped = log_path.with_name(f"{stem}_step{latest_step or 0:07d}.png")
            stable = log_path.with_suffix(".png")
            write_plot(parsed, stamped)
            # Also overwrite the stable name so `--plot` users with scripts that
            # always read <log>.png still see the freshest data.
            try:
                import shutil

                shutil.copyfile(stamped, stable)
                print(f"[inspect] also refreshed {stable}")
            except OSError as e:
                print(f"[inspect] warn: could not refresh {stable}: {e}", file=sys.stderr)

    if args.watch > 0:
        try:
            while True:
                # clear screen for watch mode
                sys.stdout.write("\x1b[2J\x1b[H")
                one_pass()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[inspect] stopped")
    else:
        one_pass()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
