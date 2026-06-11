# Milestones

A chronological log of significant integration milestones for the X2
Ultra + GR00T N1.7 + SONIC stack. Each entry is a session-level wrap-up
with the run command, headline numbers, code changes, issues
encountered + fixes applied, and what's left for the next session.

```{toctree}
:maxdepth: 1

2026-05-02_first_iter4000_powered_run
2026-05-02_post_deploy_tuning
2026-05-03_first_iter22000_powered_walk
2026-05-08_live_vla_sonic_sim_v0
2026-05-10_omnihand_finger_tuning
2026-05-11_finger_tip_oppose_signal
2026-05-12_finger_signal_filter
2026-05-10_sonic_loop_v1_schema
2026-06-07_vla_sim_stereo_safe_debug
2026-06-07_vla_bridge_heading_stability_and_head_lock
2026-06-07_wrist_offset_v1
2026-06-08_arm_freeze_on_upstream_stall
2026-06-09_vla_wire_tuning_iter
2026-06-10_vla_manual_takeover
2026-06-10_vla_closed_loop_wire
```

## At a glance

| Date | Milestone | Surface | Outcome |
|------|-----------|---------|---------|
| 2026-05-02 | First iter-4000 powered run | Real X2 + SONIC | First-ever powered policy on hardware. Tilt/balance only — no walking. |
| 2026-05-02 | Post-deploy tuning | Real X2 + SONIC | KP/KD sweep, target-LPF and clamp tuning gauntlet. |
| 2026-05-03 | First iter-22000 powered walk | Real X2 + SONIC | First out-and-back walking cycle, 36.75 s wall-time, clean MC handoff. |
| 2026-05-08 | Live VLA → SONIC sim (v0) | **Sim-only**, X2 + SONIC + N1.7 | Closed-loop VLA → SONIC pipeline runs end to end. Visible motion is mode-collapsed; full triage is documented in the milestone page. |
| 2026-05-10 | OmniHand finger-tuning iteration | **Sim-only**, Quest 3 → X2 kinematic | Thumb-fingertip-touch gesture now closes correctly. Anchor expansion + 3-motor opposition fold-in. Topology mismatch on non-thumb tips filed for v1. |
| 2026-05-11 | Per-finger fingertip-to-thumb proximity (v0.5) | **Sim-only**, Quest 3 → X2 kinematic | New JS `computeFingerTipOppose` 4-vector + Python `max(curl, finger_tip_oppose)` drive on non-thumb pips + pip CLOSED anchor 80° → 88°. Wired end-to-end (JS → reader → retargeter → debug NPZ → replay); needs a fresh recording to verify visually. |
| 2026-05-12 | Finger-signal smoothing filter (v0.6) | **Sim-only**, Quest 3 → X2 kinematic | EMA(α=0.5) + rolling-median deadband-hold on the 10 per-side hand inputs. Calibrated against v5/ep1: held-pose tremor reduced 20–40 % on the worst fingers, +20 ms motion lag, 0 ms touch-onset lag. Live + record + replay paths all wired; debug NPZ persists raw + filtered for offline A/B. |
| 2026-05-10 | SONIC-loop v1 dataset schema | **Sim-only**, X2 + SONIC 25k + Quest 3 | Canonical training-target columns flip from operator-commanded to **post-SONIC executed** q. Operator command preserved as `_pre_sonic` siblings (debug-only, training-invisible). New `meta/dataset_format_version.json` marker, `inspect_sonic_correction.py` offline diagnostic, live `--sonic-correction-warn-rad` operator log. `record_x2_dataset.sh` is now the recommended path for VLA captures. |
| 2026-06-07 | VLA sim-first debug w/ stereo cameras (M5 v1) | **Sim-only**, X2 + SONIC + real-robot fine-tune | After real-robot VLA hit ~2.77 rad joint deviations on chunk 2, pivot to debugging in sim. New `_GhostCameraProvider` + auto-promotion lets the sim path satisfy the `omnihand_stereo` modality with a single MuJoCo render of `stereo_head_front` aliased into both `stereo_left`/`stereo_right` ("degenerate stereo"). `run_live_vla_demo.sh` defaults flipped to match the real-robot launcher (decoder, ramp, LPF, TQDM disabled). |
| 2026-06-07 | VLA bridge heading stability + head-lock on real robot | **Real X2 + SONIC + N1.7** | Three bridge-side patches: (a) bootstrap-safe pose-publish gate (withhold first publish until `x2_debug` arrives, kills the phantom yaw=0 reference that dragged robot toward world +X on first VLA start), (b) live yaw-rebase on `root_quat_xyzw` + future window, (c) surgical `waist_yaw` (MJ slot 12) pin to measured during the `legs,waist` freeze so the idle_stand clip's ~33° offset stops driving steady-state heading drift. Plus `--lock-head-straight` on `x2_pc2_daemons.sh` (= `--max-target-dev-head 0.01`) to keep the head centered. |
| 2026-06-07 | Calibration palm-direction lock + multi-pose wrist offset (v1) | **Real X2 + sim**, Quest 3 stack | Lock palm direction in every calibration prompt (printed + TTS) so the arms-down `wrist_alignment_quat` reference is reproducible. New `fit.<side>.op_quat_offset_rpy_deg` field on `ArmFit` persists a per-operator wrist-quat offset in the calibration YAML; `VRArmTeleopCalibrated` resolves it three-tier (CLI > YAML > identity), so every launcher (kinematic teleop, dataset recording, VLA bridge) picks it up automatically. Recommended values for this operator (LEFT `-5.7 +1.8 -6.0`, RIGHT `+0.3 +4.4 +10.4`) derived from a Markley quaternion LS average over `arms_down + t_pose + arms_forward` (namaste excluded). Cuts the worst-case IK target rotation residual from 120° (arms-down-only fit) to ~10° per pose; pitch joint stops saturating in arms-forward. **Known limitation**: single constant offset ≈ 10° per pose; see milestone "what to revisit" section for the per-pose-interpolation follow-up. |
| 2026-06-10 | Manual takeover during VLA inference | **Real X2 + SONIC + N1.7**, Quest 3 stack | Operator can grab the wire from a stuck VLA via VR teleop without restarting the bridge. Two-part change: (a) `x2_pose_proxy.py` gains optional dual-source arbitration (`--override-port`) and an edge-triggered control PUB (`--vla-control-port`) — override frames win whenever fresh, debounced via `--override-stale-ms`. (b) `live_vla_publish_motion_token.py` gains a `vla_control` SUB and a `_VlaControlSignal` cold-restart pathway — on the override_released edge the bridge clears ramp / LPF / chunk-blend state, pins a chunk-id baseline (drops pre-override chunks), and hold-publishes the operator's measured pose for `--vla-cold-restart-hold-ticks` (default 25 = 500 ms) so the proxy's HOLD -> LIVE handoff doesn't see a step change. Quest 3 manager is **untouched** — engagement is implicit (whatever publishes on the override port wins). All new code paths gated on positive port numbers; defaults disabled, byte-for-byte unchanged on existing single-source runs. |
| 2026-06-08 | Pose-proxy fallback ladder: freeze arms on upstream stall | **Real X2 + SONIC + VR teleop** | Replaces the PC2 pose proxy's binary `LIVE/IDLE` fallback with a staged `LIVE -> HOLD -> BLEND -> IDLE_CLIP` ladder so a WiFi blip / laptop GC pause / Cursor reload during teleop no longer slams the arms into default-stand. HOLD (default 10 s) re-publishes the LAST forwarded upstream BYTES verbatim -> deploy sees `jvel=0`, zero kinematic surprise. BLEND (default 3 s) lerps `joint_pos_mj` toward the baked idle clip so the eventual fall-back is a smooth glide rather than a step. New `--idle-mode {blend,hold-last,idle-stand}` CLI knob (default `blend`); `idle-stand` reproduces pre-2026-06-08 behaviour as a regression escape. C++ deploy untouched. 20 new unit tests; 38/38 proxy tests passing. |
| 2026-06-10 | Closed-loop tracking feedback on the VLA wire | **Real X2 + SONIC + N1.7 VLA bridge** | Adds optional per-arm-joint closed-loop tracking feedback to the VLA bridge wire step cap to stop the open-loop drift / battery-sag / motor-temp oscillations that the v3 static defaults (LPF / blend / scalar step-cap) couldn't reliably suppress. New `_apply_tracking_feedback` helper observes `x2_debug`'s measured arm-joint positions + velocities each tick and per-joint-throttles the wire delta via two composed laws (position backoff on `|target - measured|`, velocity cap on `|measured_dq|`). New `_clamp_vector_step_per_joint` clamps each joint independently (vs scalar variant's whole-vector scale) so only the lagging joint slows down. Arms-only joint mask (legs/waist/head pass through); falls back to the scalar clamp when proprio is stale. **Step 1 rollout: default OFF**, opt in via `--vla-tracking-feedback` or `VLA_TRACKING_FEEDBACK=1`; v3 defaults stay in place so feedback is additive belt-and-suspenders. Step 2 (separate commit) will flip default ON and relax v3 statics. New `tf_throttle=N/14` field in pub-tick log. 30 new unit tests + 5 new launcher CLI tests; 25/25 sim-proxy + 30/30 feedback tests green. |
