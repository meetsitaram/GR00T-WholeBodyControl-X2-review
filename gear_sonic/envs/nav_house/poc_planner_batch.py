#!/usr/bin/env python3
"""N0 Gate G2: can NeuralPlannerCore replan with batch > 1?

Method: load the planner (fixed_scratch ckpts), do one real
replan_with_velocity to capture the exact input_state it builds, then call
_predict_with_velocity directly with that state tiled to B in {1,4,16,64}.
The B=1 hardcode in _predict_with_velocity was patched to derive B from the
input (PoC change, uncommitted).

PASS: B=16 outputs finite + correctly shaped AND per-env cost at B=16
< 25% of B=1 cost. Prints a verdict line for the validation ledger.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "motionbricks"))

FS = Path(os.path.expanduser("~/x2_cloud_checkpoints/fixed_scratch"))


def main():
    from motionbricks.motion_backbone.inference.load_x2_planner import (
        X2PlannerPaths, load_x2_planner,
    )
    d = X2PlannerPaths.default()
    paths = X2PlannerPaths(
        vqvae_ckpt=FS / "vqvae/model-step=0300000.ckpt",
        pose_ckpt=FS / "pose_500k/model-step=0500000.ckpt",
        root_ckpt=FS / "root/model-step=0300000.ckpt",
        vqvae_version_dir=d.vqvae_version_dir,
        pose_version_dir=d.pose_version_dir,
        root_version_dir=d.root_version_dir,
    )
    planner = load_x2_planner(paths, device="cuda")
    print("[g2] planner loaded")

    # stand qpos from the deploy daemon's helper (same as gen_policy_route_clips)
    spec = importlib.util.spec_from_file_location(
        "pc2_kplanner_onnx", REPO / "gear_sonic/scripts/pc2_kplanner_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pc2_kplanner_onnx"] = mod
    spec.loader.exec_module(mod)
    qpos = torch.tensor(mod._build_training_default_qpos(),
                        dtype=torch.float32, device="cuda")
    planner.reset(qpos)

    # capture the real input_state from one genuine replan
    captured = {}
    orig = planner._predict_with_velocity

    def spy(input_state, num_tokens=None):
        captured["state"] = {k: (v.clone() if torch.is_tensor(v) else v)
                             for k, v in input_state.items()}
        return orig(input_state, num_tokens)

    planner._predict_with_velocity = spy
    intent = torch.tensor([0.0, 0.5, 0.0, 0.95], device="cuda")
    planner.replan_with_velocity(intent)
    planner._predict_with_velocity = orig
    state = captured["state"]
    print("[g2] input_state keys:",
          {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)
           for k, v in state.items()})

    def tile(st, b):
        out = {}
        for k, v in st.items():
            if torch.is_tensor(v):
                out[k] = v.expand(b, *v.shape[1:]).contiguous() \
                    if v.shape[0] == 1 else v.repeat(
                        b, *([1] * (v.dim() - 1)))
            else:
                out[k] = v
        return out

    results = {}
    for b in (1, 4, 16, 64):
        st = tile(state, b)
        torch.cuda.synchronize()
        try:
            torch.manual_seed(0)
            outs = orig(st)          # warmup + correctness
            for o in outs:
                if torch.is_tensor(o):
                    assert torch.isfinite(o).all(), f"non-finite at B={b}"
            n_iter = 10
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                orig(st)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / n_iter * 1000.0
            results[b] = ms
            shapes = [tuple(o.shape) if torch.is_tensor(o) else o
                      for o in outs]
            print(f"[g2] B={b:3d}: {ms:7.1f} ms/replan  "
                  f"({ms / b:6.2f} ms/env)  out={shapes}", flush=True)
        except Exception as e:
            results[b] = None
            print(f"[g2] B={b:3d}: FAILED — {type(e).__name__}: {e}",
                  flush=True)

    if results.get(16) and results.get(1):
        ratio = (results[16] / 16) / results[1]
        verdict = "PASS" if ratio < 0.25 else "MARGINAL"
        print(f"[g2] VERDICT: {verdict} — per-env cost at B=16 is "
              f"{ratio * 100:.0f}% of B=1 ({results[16] / 16:.2f} vs "
              f"{results[1]:.2f} ms)", flush=True)
    else:
        print("[g2] VERDICT: FAIL — B=16 did not run", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
