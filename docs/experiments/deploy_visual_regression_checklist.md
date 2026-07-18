# Deploy Visual Regression Checklist

**Run this in SIM before shipping ANY new sonic model or kplanner code/model to the robot.**
Establishes a fixed, repeatable set of motions to eyeball so a deploy can't silently regress
walking, turning, dances, or combat. Introduced 2026-07-18 after the kplanner handoff fix — the
lesson: model/code changes can pass numeric gates yet break motion, so a standing *visual* gate
is required.

## Why both parts

- **Part A (kplanner + handoff)** exercises the planner→sonic pipeline live (the 30→50 Hz
  resample+blend, replan seams, idle↔walk transitions). A numeric parity PASS on the ONNX graph
  does NOT prove the live handoff is clean.
- **Part B (sonic tracking)** exercises the deploy model's tracking on a fixed reference set
  spanning the motion classes we care about (locomotion, styled dance, combat).

A deploy is GO only if **every** item below is visually clean.

## Part A — kplanner-driven (planner + handoff)

Launch the pad sim with the model under test (walk mode), drive with **L2 + left stick**:

```bash
# torch FT trio (or swap --kplanner-*-ckpt / --model for the model under test)
KPLANNER_FIXED_FWD_MPS=0.5 ./gear_sonic/scripts/sim_onnx_planner.sh   # or /tmp/launch_sim_walk05.sh
```

| # | motion | how | PASS criteria |
|---|---|---|---|
| A1 | slow straight walk | L2 + stick fwd, ~0.5 m/s | natural cadence, **no foot skate/slip**, arms swing, no drift/spin |
| A2 | regular walk + turn | fwd + right-stick X (or angle stick) | turns follow command, feet plant cleanly through the turn, no stumble |
| A3 | idle↔walk transition | release/re-press L2 | no arm-pose snap, no lurch, holds heading (orientation stable) |

*(If foot skate appears, suspect the handoff/fps path — grep the runtime for
`get_next_frame_resampled`. See `docs/experiments/kplanner_sonic_handoff_g1_parity.md`.)*

## PC2 identity gate (runs first, required)

`deploy_regression_check.sh` **requires `--pc2 <ip>`**. Before showing a single clip it
md5-compares what you are testing against what the robot actually runs:

| check | what it proves |
|---|---|
| sonic | the deployed `agibot_x2_sonic.onnx` == the ONNX form of the model under test |
| planner template/velocity | the deployed planner graphs == your local graphs |
| handoff fix | the robot's `pc2_kplanner_onnx.py` contains `get_next_frame_resampled` |

Verdict is printed as **GATE MATCHED** (results certify the robot's build) or
**GATE MISMATCHED** (they do not). Use `--no-pc2` *only* when validating a candidate model
*before* deploying it — a mismatch is expected there.

> **Why this exists:** on its first run the gate caught that the robot was running
> **`ft_2082_g1.onnx`** while every sim launcher used **`walkft_3065_g1.onnx`** — i.e. months
> of sim testing were against a *different sonic than the robot*. A model can pass every
> numeric and visual gate locally and still not be the model on the robot. Never certify a
> deploy from a MISMATCHED run.

## Part B — sonic tracking (deploy model)

```bash
./gear_sonic/scripts/deploy_regression_check.sh --pc2 192.168.86.32            # robot's current model
./gear_sonic/scripts/deploy_regression_check.sh --pc2 10.0.1.41 <model.pt>     # specific model
./gear_sonic/scripts/deploy_regression_check.sh --no-pc2 <candidate.pt>        # pre-deploy candidate
# viewer: N=next clip, SPACE=pause, ,/. = speed
```

Curated set (`gear_sonic/data/motions/deploy_regression_suite.pkl`, 7 clips):

| # | class | clip | PASS criteria |
|---|---|---|---|
| B1 | walk | `slow_walk_0.5_001` | tracks the walk, no fall, no skate |
| B2 | walk+turn | `slow_walk_turns_0.5_001` | follows the turning reference, stays upright |
| B3 | easy dance | `dance_party_hips_003__A467` | reproduces hip motion, hands/arms track, stable |
| B4 | easy dance | `dance_freedom_wheels_001__A465` | arm/body motion faithful, no fall |
| B5 | medium dance | `dance_disco_fever_001__A465` | more dynamic — no fall, feet don't cross/skate |
| B6 | medium dance | `dance_hiphop_funky_guitar_R_fast_001__A319` | fast arms tracked, base stable |
| B7 | combat/boxing | `ROM_Box_01_Box_02_..._A520` | punch/guard poses reproduced, no topple |

## Verdict

- **GO** — all A1–A3 and B1–B7 visually clean.
- **NO-GO** — any fall, sustained foot skate, failure to reproduce the motion, or an
  orientation/arm-snap. Fix and re-run the full suite; do not partial-ship.

Record the run (model, date, any per-clip notes) below when you deploy.

### Deploy log

| date | model / code | verdict | notes |
|---|---|---|---|
| _template_ | _e.g. softland_4800 + handoff-fix_ | GO / NO-GO | _per-clip observations_ |
