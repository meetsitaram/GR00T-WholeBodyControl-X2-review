# 2026-06-07 — VLA bridge heading stability + head-lock on real robot

> **Session focus.** Three real-robot tuning patches that turn the
> autonomous VLA path from "always turns the robot toward world yaw=0
> on first start, head drifts off-center under SONIC" into "starts
> heading-agnostic, head stays straight ahead." All fixes are on the
> bridge / launcher side; the C++ deploy is untouched.

---

## TL;DR

| Symptom (before) | Cause | Fix |
|---|---|---|
| First VLA start of a SONIC session yanks the robot toward world yaw=0 (the heading at SONIC boot). At -180° starting offset, robot spins ~180°. Subsequent runs hold heading. VR teleop is unaffected. | Bridge publishes `root_quat_xyzw = identity` (yaw=0) during the bootstrap window between PUB-bind and the first `x2_debug` arrival. The PC2 deploy is already in CONTROL and locks onto that phantom reference — the deploy's bootstrap-safe `measured_quat` override only fires while `LastReceivedMonotonicS() < 0`, which the bridge was prematurely satisfying. | Withhold pose publish until `state.received_any` flips True (first `x2_debug` frame). |
| With root_quat fixed, robot still slowly drifts in heading after VLA goes live, and `manipulation`-mode freeze on `legs,waist` to `idle_stand` drove a ~33° steady-state waist_yaw offset (clip's frame-0 waist_yaw is ~0.58 rad off DEFAULT_STAND_POSE). | `_apply_frozen_body_groups` overwrites all frozen DOFs with the idle clip, including `waist_yaw_joint` (MJ slot 12, the dominant heading-correction effector). | Pin **only** `waist_yaw` (slot 12) to the live measured value; keep clip jitter on every other frozen DOF so the policy's training-distribution balance signal is preserved. |
| With SONIC running, the head feels stiff and locked to ~+20° (cmd +0.50 rad, measured +0.345 rad). VR / VLA wires can't override head joints because there is no head bypass (unlike wrists). | The policy's head target drifts off-center, and the C++ deploy holds it stiffly via the per-joint PD on `/aima/hal/joint/head/command`. The tuning YAML's `max_target_dev_head` defaults to `0.50` rad (~29°), which is too loose to keep the head centered. | Add `--lock-head-straight` to `x2_pc2_daemons.sh`. Expands to `--max-target-dev-head 0.01` on `deploy_x2.sh`, clamping the policy's head yaw target within ±0.6° of the trained default (yaw=0). |

---

## What landed

### Bridge — `gear_sonic/scripts/live_vla_publish_motion_token.py`

1. **Live yaw-rebase on `root_quat_xyzw`** — `_root_quat_xyzw_from_base_quat_wxyz` extracts yaw from the latest `x2_debug` `base_quat`, builds `R_z(yaw)`, and overrides both `cur_quat` and `root_quat_xyzw_future` (all 9 slots) on every publish. Mirrors `X2DatasetRecorder._compute_idle_root_quat_xyzw`.
2. **Bootstrap-safe publish gate** — `if not silent_wire and state.received_any: pub_sock.send(...)`. Sticky one-way; the gate stays open once tripped. Adds two one-shot log lines for visibility:
   ```
   [live-VLA] withholding pose publish until first x2_debug frame arrives ...
   [live-VLA] first pose publish (tick=N); x2_debug seen, root_quat now tracks live heading.
   ```
3. **Surgical `waist_yaw` pin during freeze** — Section F keeps the existing `idle_baseline` freeze (clip jitter required for balance), then overrides `cur_jpos[WAIST_YAW_IDX]` and `joint_pos_mj_future[:, WAIST_YAW_IDX]` with `body_q_mj_now[WAIST_YAW_IDX]` whenever `deploy_fresh` AND waist_yaw is in the freeze set. `joint_vel_mj_future` is recomputed from the patched positions.

### PC2 daemons — `gear_sonic_deploy/scripts/x2_pc2_daemons.sh`

Convenience flag `--lock-head-straight`:

```sh
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start ... --lock-head-straight
```

Expands to `--max-target-dev-head ${LOCK_HEAD_STRAIGHT_RAD:-0.01}` on the deploy. Both `gesture_commands.md` and `pick_place_commands.md` now include it in the canonical start command.

### Docs

- `pick_place_commands.md` — operator-facing sections on `--lock-head-straight` (rationale + table of what it does and doesn't do) and "Heading stability on VLA start" with bridge-log markers.
- `gesture_commands.md` — same head-lock rationale.
- `docs/source/tutorials/x2_vla_runtime.md` — new troubleshooting section "Robot turns sharply toward 'world yaw=0' on first VLA start" enumerating the three bridge fixes and the log lines to grep for.

---

## How to confirm the heading fix

Run the bridge from progressively more-extreme starting headings.
Expected behavior:

| Starting heading | Before | After |
|---|---|---|
| -6.7° (close to neutral) | small right turn back to ~0° | holds heading |
| -45° | ~45° right turn back to ~0° | holds heading |
| ±180° | ~180° rotation back to ~0° | holds heading |

In `${RUN_DIR}/bridge.log`, look for the three sentinel lines in this
exact order:

```text
[live-VLA] withholding pose publish until first x2_debug frame arrives ...
[live-VLA] first pose publish (tick=N); x2_debug seen, root_quat now tracks live heading.
[live-VLA] root_quat yaw-rebase ACTIVE: ... (yaw=+178.2deg)
```

If you see "first pose publish" *before* `x2_debug` has been seen (or
if the `yaw-rebase ACTIVE` log is missing entirely), the bootstrap
race is back.

## How to confirm the lower-body stability fix

In `manipulation` mode (the default), confirm the lower body holds
upright through a 5+ min idle without drift:

```sh
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
    --motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "grab the can from the table"
```

Watch `${RUN_DIR}/x2_debug_summary.json` after the run:

- `min_grav_z` should stay below ~-0.97 (upright); a value drifting toward ~-0.85 indicates the robot is leaning >25° and the clip-jitter balance signal isn't reaching the policy.
- `tilt_trip_count` should be 0.
- `max_body_q_drift` should stay below ~0.6 rad (the action-clip threshold).

## Past failure modes the new fixes do **not** address

- **Decoder fed bad proprio** — separate diagnostic; see milestone
  [2026-06-07 VLA sim-first debug w/ stereo cameras](2026-06-07_vla_sim_stereo_safe_debug.md).
- **Inference period blow-up under GPU throttling** — addressed by
  `INFERENCE_MIN_PERIOD_S=1.5` default; not heading-related.
- **Head pitch up/down control** — firmware limitation; head pitch
  motor is not actuated. `--lock-head-straight` only affects head
  yaw.

## Open work

- Add a runtime test that intentionally rotates the robot to ±90° in
  the gantry before starting VLA and asserts `bridge.log` carries the
  three sentinel lines plus a near-zero heading delta in the first
  10 s of `x2_debug_trace.csv`.
- Consider unifying the bridge's `_compute_idle_root_quat_xyzw` and
  the recorder's identically-named helper into a single shared module
  to remove the duplication.
