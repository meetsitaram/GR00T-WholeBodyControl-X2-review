# SONIC-loop v1 dataset schema

**Date:** 2026-05-10  &nbsp;|&nbsp;  **Surface:** Sim-only, X2 + SONIC 25k + Quest 3

Switched the canonical training-target columns in the X2 LeRobot
dataset from operator-commanded q to **post-SONIC executed q** (what
the trained tracking policy actually achieved, i.e. what's visible
in the MuJoCo viewer). The operator's pre-SONIC X2 joint command is
preserved as `_pre_sonic` siblings for retargeter analysis but is
training-invisible. Promoted [`record_x2_dataset.sh`](../../../../gear_sonic/scripts/record_x2_dataset.sh)
as the recommended path for v1+ VLA dataset captures.

## Why

The kinematic-only teleop path
([`teleop_x2_kinematic.py`](../../../../gear_sonic/scripts/teleop_x2_kinematic.py))
pins the lower body to the stand pose and writes operator-commanded
q straight into MuJoCo. Nothing in that loop knows about feasibility,
so the operator can drive the robot through joint-limit violations
and self-collisions; the resulting parquet then contains physically
impossible action labels that a VLA fine-tune would happily learn.

The SONIC-loop recorder
([`record_x2_dataset.sh`](../../../../gear_sonic/scripts/record_x2_dataset.sh))
co-launches the C++ deploy with the trained 25k tracking policy.
The deploy treats `joint_pos_mj` as a reference, but the policy
follows it under learned reward shaping for joint limits, ground
contact, and balance. Anything physically unreachable just doesn't
get tracked, and the gap between commanded and observed q is
exactly the SONIC corrective signal we want recorded.

## v1 column layout

| Canonical (training target, post-SONIC executed q) | Debug-only (pre-SONIC, X2 joint targets) |
|---|---|
| `action.body_q_mj` | `action.body_q_mj_pre_sonic` |
| `action.left_hand_joints` | `action.left_hand_joints_pre_sonic` |
| `action.right_hand_joints` | `action.right_hand_joints_pre_sonic` |
| `action.sonic_correction_max_rad` (per-frame max arm `\|debug − canonical\|`, debug-only) | — |

Both columns in every row carry the **same vector space** — X2 joint
positions in mj-rad. The difference is *when* in the pipeline the
snapshot was taken. Only the bare-canonical columns are surfaced as
training targets via
[`get_modality_config_x2_vla`](../../../../gear_sonic/data/features_x2_vla.py);
the `_pre_sonic` siblings + correction scalar live on disk for
retargeter / SONIC-correction analysis but are never pulled into
training batches.

A new `meta/dataset_format_version.json` marker disambiguates v0
(legacy) / v1-SONIC / v1-kinematic datasets so downstream tools can
dispatch on it:

```json
{ "version": 1, "post_sonic_canonical": true,  "writer": "X2DatasetRecorder" }
{ "version": 1, "post_sonic_canonical": false, "writer": "teleop_x2_kinematic" }
```

## Code surface

* **Recorder** —
  [`gear_sonic/utils/teleop/x2_dataset_recorder.py`](../../../../gear_sonic/utils/teleop/x2_dataset_recorder.py).
  `_record_frame` now writes the post-SONIC snapshot as the canonical
  columns and the operator command as the `_pre_sonic` siblings;
  `_maybe_log_sonic_correction` prints a once-per-second arm-joint
  delta when SONIC overrides exceed `--sonic-correction-warn-rad`
  (default 0.05 rad ≈ 2.9°); `_ensure_exporter` writes the version
  marker to `meta/dataset_format_version.json`.
* **Schema** — [`gear_sonic/data/features_x2_vla.py`](../../../../gear_sonic/data/features_x2_vla.py).
  `get_features_x2_vla(post_sonic_canonical=True)` (default) returns
  the SONIC-loop schema with the four debug-only columns;
  `post_sonic_canonical=False` returns the kinematic-only schema
  without them. Modality config unchanged (it referenced
  `action.left_hand_joints` etc. by their bare names, so the rename
  is transparent to the trainer).
* **Kinematic teleop** — [`teleop_x2_kinematic.py`](../../../../gear_sonic/scripts/teleop_x2_kinematic.py)
  passes `post_sonic_canonical=False`, writes `action.body_q_mj` (=
  commanded q in this mode), and stamps a kinematic version marker.
* **Replay** — [`replay_x2_kinematic.py`](../../../../gear_sonic/scripts/replay_x2_kinematic.py)
  prefers `action.body_q_mj` and falls back to
  `action.commanded_body_q_mj` for v0 datasets.
* **Diagnostic** — new
  [`gear_sonic/scripts/inspect_sonic_correction.py`](../../../../gear_sonic/scripts/inspect_sonic_correction.py)
  prints per-arm-joint mean / p99 / max `|delta_q|` and writes a
  4-panel time-series PNG to `<dataset>/debug/sonic_correction_ep<N>.png`.
  Detects v0 / v1-SONIC / v1-kinematic via the version marker.
* **CLI** — [`record_x2_dataset.py`](../../../../gear_sonic/scripts/record_x2_dataset.py)
  gains `--sonic-correction-warn-rad FLOAT` and
  `--no-sonic-correction-log`. The wrapper
  [`record_x2_dataset.sh`](../../../../gear_sonic/scripts/record_x2_dataset.sh)
  forwards both verbatim.

## v0 → v1 migration

The `action.left_hand_joints` / `action.right_hand_joints` columns
**flip semantics** under the same name (commanded → executed).
Mixing v0 and v1 datasets in a single training run silently teaches
the model on inconsistent data. The `meta/dataset_format_version.json`
marker is the only safe dispatch.

| | v0 | v1 |
|---|---|---|
| Body action column | `action.commanded_body_q_mj` | `action.body_q_mj` |
| Body action semantics | Pre-SONIC operator command | Post-SONIC executed q |
| Hand action column names | `action.left_hand_joints` / `action.right_hand_joints` | Same names |
| Hand action semantics | Pre-deploy retarget output | Post-deploy URDF-clipped |
| `meta/dataset_format_version.json` | absent | `{"version": 1, ...}` |

## Verification

* `tests/test_record_x2_dataset_schema.py` — schema assertions
  for both `post_sonic_canonical=True` and `False`.
* `tests/test_replay_x2_kinematic.py` — fixture parquets in both
  v1 (`action.body_q_mj`) and v0 (`action.commanded_body_q_mj`)
  layouts; replay must consume either.
* End-to-end teleop_only smoke (operator-driven) is the next step
  before recording v1 episode 0.

## Next steps

1. Smoke-test `bash gear_sonic/scripts/record_x2_dataset.sh
   --teleop-only --sonic-checkpoint .../h200-iter-25000-...pt`
   with the v0.6 finger filter changes (deferred to live operator).
2. Capture v1 episode 0 to `data/lerobot/x2_quest3_sonic_v1`.
3. Run `inspect_sonic_correction.py` and confirm:
   * p99 arm delta < 0.05 rad on a calm wave;
   * a deliberate "drive arm into torso" test produces a visible
     spike in `action.sonic_correction_max_rad`;
   * `replay_x2_kinematic.py` plays the post-SONIC trajectory cleanly.
