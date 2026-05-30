# X2 kinematic-planner evaluation harness

This page documents the evaluation tools used to validate the X2 (and
G1 reference) MotionBricks kinematic planner — the same stack that
powers ``gear_sonic/scripts/x2_kplanner.py``. The harness was built
during the planner-train branch hunt that uncovered two bugs preventing
the robot from walking forward when commanded forward intent; both are
fixed (see commit ``46bd017``).

The tools fall into four layers:

1. **Per-model isolation** (``test_root_isolated.py``) — drive just the
   root model with a single replan call and read out its prediction.
2. **End-to-end velocity tracking** (``test_e2e_velocity_tracking.py``)
   — drive the full planner (VQVAE + Pose + Root + qpos integration)
   under a swept velocity grid and measure tracking ratios.
3. **Side-by-side parity** (``compare_e2e_reports.py``) — pair two JSON
   reports (typically X2 vs G1) and emit a dimensionless parity table.
4. **Visualizers** (``view_e2e_x2_vs_g1.py`` + ``run_scripted_demo.py``)
   — render the X2 stack next to the NVIDIA G1 reference stack in a
   single MuJoCo scene, either replaying a sweep grid or a scripted
   teleop sequence.

A diagnostic ``probe_root_constraint_modes.py`` script lives next to
these tools and demonstrates the original constraint-masking bug; it
documents the root cause in code form.

## Channel convention

All velocity intents are 4-D vectors ``[yaw_rate, vel_x, vel_z,
hip_h]``. Channel semantics (verified against the X2 + G1 mujoco
converters):

| channel | motion-rep axis | mujoco body axis | meaning |
|---|---|---|---|
| ``yaw_rate`` | rad/s | rad/s | rotation rate about world up |
| ``vel_x`` | X | Y | **lateral** body-frame velocity (positive = robot's LEFT) |
| ``vel_z`` | Z | X | **forward** body-frame velocity (positive = robot's FORWARD) |
| ``hip_h`` | Y | Z | hip height (m) |

Earlier docs incorrectly called ``vel_z`` lateral. The mujoco-to-motion
rotation maps mujoco-X (mjcf forward) onto motion-Z, so
``velocity_intent[2]`` is what walks the robot forward.

## Fixtures

X2 fixtures live in
``gear_sonic/data/motions/x2_ultra_locowalk.pkl`` (joblib payload, ~5 s
clips at 30 fps). The default walking fixture is
``Loop_Forward_Walk_001__A018`` (174 frames forward locomotion); the
default stationary fixture is ``Idle_Right_001__A019``.

G1 fixtures come from ``motionbricks/out/G1-clip.ckpt`` (Git LFS, ~7.5
MB). Clip 11 is a forward-walk sequence and clip 0 is a stationary
hold.

## Quick start

Run the full end-to-end sweep on X2 and the G1 reference, then build a
parity table:

```bash
cd <repo_root>
source .venv/bin/activate
export PYTHONPATH="${PWD}/motionbricks:${PWD}"

python motionbricks/scripts/test_e2e_velocity_tracking.py \
  --ckpt-set x2 --fixture walking --sweep all --horizon-s 3.0 \
  --save-npz out/per_model_report/e2e_x2_all.npz \
  --report-json out/per_model_report/e2e_x2_all.json

python motionbricks/scripts/test_e2e_velocity_tracking.py \
  --ckpt-set g1 --fixture walking --sweep all --horizon-s 3.0 \
  --save-npz out/per_model_report/e2e_g1_all.npz \
  --report-json out/per_model_report/e2e_g1_all.json

python motionbricks/scripts/compare_e2e_reports.py \
  --left  out/per_model_report/e2e_x2_all.json --left-label X2 \
  --right out/per_model_report/e2e_g1_all.json --right-label G1
```

Visualize side-by-side:

```bash
python motionbricks/scripts/view_e2e_x2_vs_g1.py \
  --x2-npz out/per_model_report/e2e_x2_all.npz \
  --g1-npz out/per_model_report/e2e_g1_all.npz
```

Generate a scripted teleop demo (walk → run → in-place turns → walking
arcs → walk back → side-step) and view side-by-side:

```bash
python motionbricks/scripts/run_scripted_demo.py --ckpt-set x2 \
  --save-npz out/per_model_report/demo_x2.npz
python motionbricks/scripts/run_scripted_demo.py --ckpt-set g1 \
  --save-npz out/per_model_report/demo_g1.npz
python motionbricks/scripts/view_e2e_x2_vs_g1.py \
  --x2-npz out/per_model_report/demo_x2.npz \
  --g1-npz out/per_model_report/demo_g1.npz
```

## Metrics

Two layers, deliberately:

**Physical (in meters / degrees)** — diagnostic and human-readable.

- ``achieved_forward_m``, ``achieved_lateral_m``: body-frame XY
  displacement over the horizon (projected back through the initial
  yaw).
- ``achieved_dyaw_deg``: total yaw change over the horizon.
- ``hip_z_mean_m`` / ``hip_z_std_m``: hip height mean and stdev (fall
  alarm if std blows up).
- ``lateral_leakage`` = ``|lat| / |fwd|`` for forward trials —
  sideways-drift fraction.

**Dimensionless (ratios)** — cross-skeleton comparable, target ~1.0.

- ``tracking_forward`` = ``achieved_fwd / (vel_z_cmd * horizon_s)``
- ``tracking_lateral`` = ``achieved_lat / (vel_x_cmd * horizon_s)``
- ``tracking_yaw`` = ``achieved_dyaw_rad / (yaw_rate_cmd * horizon_s)``
- ``slope_<axis>`` = least-squares slope of achieved vs commanded across
  the sweep grid, normalized by ``horizon_s``.

A slope of 1.0 means perfect velocity tracking. < 1 = under-tracking;
> 1 = over-tracking.

## Known findings

These are the X2-vs-G1 results from the sweep + scripted-demo runs
(post-bugfix). All slopes ideal at 1.0.

| axis | X2 slope | G1 slope | verdict |
|---|---|---|---|
| forward | 1.10 | 1.25 | PARITY — both over-track ~10–25% |
| lateral | 1.01 | 1.54 | X2 closer to ideal; G1 over-shoots large side-steps |
| yaw | 2.35 | 1.27 | G1 closer; X2 over-rotates ~2× |

### Per-axis observations

- **Forward locomotion** is at structural parity with G1. Both stacks
  over-track ~15% on average. Robot walks the right direction with the
  right magnitude.
- **X2 lateral** is actually *better* than the G1 reference. X2 hits
  ratio 0.89–1.47 across the symmetric ±0.30 m/s grid; G1 hits
  0.91–2.15.
- **X2 yaw** is the one regression worth tracking. It over-rotates by
  2.0–2.9× across the ±0.4 rad/s grid. G1 stays 0.95–1.34. Likely
  culprits: X2 training data is locowalk-only with limited turn
  examples, and the ``target_global_heading`` constraint is still
  masked off (the bugfix only unmasked ``target_global_root_values``).
- **Hip-Z stability**: X2 hip-Z std is 0.15–1.58 cm; G1 is 0.62–4.66 cm.
  X2 plants its feet more cleanly than the G1 reference.

### Training-data quirks (not planner bugs)

- **X2 walks with stiff knees** (~6° mean flex, ~30° peak) because the
  ``x2_ultra_locowalk.pkl`` training clips were generated by a
  stiff-legged locomotion policy. The model faithfully reproduces this
  distribution. G1's NVIDIA-trained model bends knees ~45° mean / ~110°
  peak because its training data includes mocap-retargeted walks.
- **G1 walks with arms extended forward** — characteristic of the G1
  demo dataset's manipulation/carry examples. X2 (locowalk-only) keeps
  arms naturally relaxed.

Both are useful **sanity checks** that the planner is consuming the
right codebook and normalization stats: each stack is producing its own
training-data pose distribution, not a mixed or scrambled one.

## Script reference

| script | purpose |
|---|---|
| ``test_root_isolated.py`` | One replan; reads back root-model output + integrated qpos. Supports ``--mode {single, cold_sweep, warm_sweep}``. |
| ``test_e2e_velocity_tracking.py`` | Full-stack velocity sweep with two-layer metrics. Supports ``--sweep {forward, lateral, yaw, all}``. |
| ``compare_e2e_reports.py`` | Pairs two JSON reports and prints a parity table + verdict per axis. |
| ``probe_root_constraint_modes.py`` | Diagnostic that originally proved bug 1; runs the root model under three constraint-masking modes (``velocity_only``, ``velocity_plus_target_pos``, ``demo_full``). |
| ``plot_root_isolated_sweep.py`` | Plots forward displacement vs commanded velocity from ``test_root_isolated.py`` JSON. |
| ``view_e2e_x2_vs_g1.py`` | Dual-skeleton MuJoCo viewer (X2 + G1 in one scene, canonicalized to start at (0, ±sep/2) facing +X). Plays sweep grids or scripted demos. |
| ``run_scripted_demo.py`` | Drives the planner through a hardcoded sequence (slow walk → run → stop → in-place turns → walking arcs → walk back → side-step). Output NPZ is viewer-compatible. |
| ``load_g1_planner.py`` *(module)* | Loads the NVIDIA G1 MotionBricks checkpoints into ``NeuralPlannerCore``. Parallel to ``load_x2_planner.py``. |

## Bugfix history

Two bugs that together prevented the planner from walking forward when
commanded forward intent (both fixed in ``46bd017``):

1. **``NeuralPlannerCore._predict_with_velocity`` masked the
   target-position constraint**. Models were trained with
   ``target_global_root_values`` observable on the last NUM_FT frames
   of the constraint window, but the planner was masking that constraint
   off. With the mask in place the root model under-tracked velocity by
   ~20×. Fix: compute an implied target position
   ``last_context_pos + velocity_intent * target_horizon_s`` and unmask
   the constraint. Diagnostic slope went from 0.07 to 0.95 on both X2
   and G1.

2. **Velocity channel swap in ``gear_sonic/scripts/x2_kplanner.py`` and
   ``motionbricks/scripts/replay_pkl_through_kplanner.py``**. The
   ``_BASE_VELOCITY`` tables and rolling-window intent extraction routed
   forward speed into ``vel_x`` (= lateral channel) and side-step into
   ``vel_z`` (= forward channel). End result: forward commands made the
   robot side-step, lateral commands stalled. Fix: swap the channel
   assignments and correct the docstrings.

End-to-end verification after both fixes: ``replay_pkl_through_kplanner.py``
with ``vel_z=0.45`` on ``Loop_Forward_Walk_001__A018`` produces
``pred dy_m = -2.325`` vs ``actual dy_m = -2.597`` (10% error, correct
direction). Side-by-side MuJoCo viewer confirms the predicted robot
walks forward in lock-step with the source clip.
