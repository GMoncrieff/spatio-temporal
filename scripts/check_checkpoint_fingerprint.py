#!/usr/bin/env python3
"""Prove a checkpoint is the shipped architecture before spending hours predicting with it.

"No extra flags" is argparse defaults, not the production architecture: ``--central_residual``
defaults to False, under which the central head predicts absolute HM and the head takes
``hidden_dim`` channels instead of ``hidden_dim + N_CONTEXT_CHANNELS``. A run on the wrong
architecture either dies with a size mismatch or, worse, trains happily and scores badly for
reasons that read like a data problem.

The check is the central head's input width: 72 = hidden_dim 64 + the 8 change-context
channels. Pass ``--control`` with a checkpoint you know is wrong; if it is not REJECTED the
check is not discriminating and its verdict on the real checkpoints means nothing.

    python scripts/check_checkpoint_fingerprint.py --manifest data/conv_update/PROVENANCE.json \
        --control artifacts/model-khrpthgy:v0/model.ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

CENTRAL_HEAD_W = "model.central_heads.0.0.weight"
QUANTILE_HEAD_W = "model.lower_heads.0.0.weight"


def fingerprint(path: str) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    if CENTRAL_HEAD_W not in sd:
        raise KeyError(f"{path}: no {CENTRAL_HEAD_W} -- not a SpatioTemporalPredictor checkpoint")
    out_ch, central_in, _, _ = sd[CENTRAL_HEAD_W].shape
    quantile_in = sd[QUANTILE_HEAD_W].shape[1] if QUANTILE_HEAD_W in sd else None
    return {
        "hidden_dim": int(out_ch),
        "central_head_in": int(central_in),
        "quantile_head_in": int(quantile_in) if quantile_in is not None else None,
        "context_channels": int(central_in - out_ch),
        "n_horizon_heads": sum(1 for k in sd if k.startswith("model.central_heads.") and k.endswith(".0.weight")),
    }


def check(path: str, context_channels: int) -> tuple[bool, dict | str]:
    try:
        fp = fingerprint(path)
    except Exception as e:  # a checkpoint that will not even load is a rejection
        return False, f"{type(e).__name__}: {e}"
    return fp["context_channels"] == context_channels, fp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="JSON with 'checkpoints' {fold: path} and 'production_model'")
    ap.add_argument("--checkpoint", action="append", default=[], help="repeatable explicit path")
    ap.add_argument("--control", help="a checkpoint that MUST be rejected")
    ap.add_argument("--context_channels", type=int, default=8,
                    help="expected central_head_in - hidden_dim (default 8)")
    args = ap.parse_args(argv)

    targets = []
    if args.manifest:
        m = json.loads(Path(args.manifest).read_text())
        for fold, p in sorted(m.get("checkpoints", {}).items()):
            targets.append((f"fold{fold}", p))
        if m.get("production_model"):
            targets.append(("production", m["production_model"]))
    targets += [(Path(p).parent.parent.name, p) for p in args.checkpoint]
    if not targets:
        raise SystemExit("nothing to check: pass --manifest or --checkpoint")

    failures = 0
    for name, p in targets:
        ok, fp = check(p, args.context_channels)
        if ok:
            print(f"  ✓ {name:<12} head_in {fp['central_head_in']} "
                  f"(hidden {fp['hidden_dim']} + context {fp['context_channels']}), "
                  f"quantile_in {fp['quantile_head_in']}, {fp['n_horizon_heads']} horizon heads")
        else:
            failures += 1
            print(f"  ✗ {name:<12} REJECTED: {fp}")

    if args.control is None:
        print("\n⚠ no --control given: this check has not been shown to reject anything")
        return 1 if failures else 0

    ok, fp = check(args.control, args.context_channels)
    if ok:
        print(f"\n✗ control {args.control} was ACCEPTED -- the check is not discriminating")
        return 2
    print(f"\n  ✓ control  {args.control} correctly REJECTED: {fp}")

    if failures:
        print(f"\n✗ {failures} checkpoint(s) are not the shipped architecture")
        return 1
    print(f"\n✓ all {len(targets)} checkpoints are the shipped architecture, control rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
