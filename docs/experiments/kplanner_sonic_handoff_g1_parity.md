# Kplanner → SONIC Handoff: G1 Parity Fix (30 Hz → 50 Hz Resample + Blend)

**Date:** 2026-07-17
**Severity:** fundamental — affected every live kplanner-driven walk/run since the X2 deploy path was written
**One-line:** The X2 kplanner→sonic publish path never implemented the G1 stock stack's 30 Hz→50 Hz output resampling and 8-frame cross-fade blend, so it played the 30 fps planner motion **1.67× too fast** with discontinuous replan seams — causing foot slippage, hurried cadence, and the long-standing start-slip / tripping artifacts.

---

## Symptoms (what we kept seeing)

- Live gamepad drive: **foot slippage**, hurried/"skating" gait, footsteps that didn't match the reference.
- Start-from-idle foot slips, occasional forward tripping (historical).
- The **exact same** kplanner output, replayed **offline**, looked clean — so it "felt like" a model problem but wasn't.

## The isolation (planner vs sonic vs handoff)

We split the pipeline into independently-viewable stages against a recorded reference clip (`slow_walk_0.5_001` from `x2_g1teleop_30fps.pkl`):

1. **Planner alone** (`replay_pkl_through_kplanner.py` → `view_kplanner_replay.py`): feed the clip's velocity/facing to the FT trio, view the kinematic output. → **Clean gait**, plausible poses, good arm swing (the model *adds* arm swing the stiff recording lacked). Root drifts laterally (~2 m) — a separate, milder root-model weakness.
2. **SONIC tracking the raw kplanner output** (`eval_x2_mujoco.py --motions <kplanner-output-pkl>`): → **looked great** — sonic reproduced the walk with natural arm swing.
3. **Live stack** (kplanner→sonic, continuous, gamepad): → **foot slippage, wrong cadence.**

Stages 1–2 good, stage 3 bad ⇒ the fault is in the **live handoff**, not the models.

## Root cause

The kplanner model outputs motion at **30 fps** (`load_x2_planner(...).fps == 30.0`). The live publish loop runs the control/output at **50 Hz** (`OUTPUT_FPS = 50`) and consumed the ring buffer by advancing **one whole frame per tick**:

```
NeuralPlannerCore.get_next_frame(): self._current_frame_idx = min(idx + 1, ...)
x2_kplanner publish loop: period_s = 1/OUTPUT_FPS (50 Hz), calls get_next_frame() once/tick
```

30 fps content consumed at 50 frames/s = **50/30 = 1.67× too fast.** Measured on `slow_walk_0.5_001` (pred travels 3.81 m / 295 frames, gait self-consistent at stride 0.476 m/step):

| played at | duration | forward speed |
|---|---|---|
| 30 fps (native) | 9.8 s | 0.388 m/s (looks slow-mo; root model slightly weak) |
| **50 Hz (the bug)** | 5.9 s | **0.646 m/s** (too fast; overshoots the 0.5 command) |
| correct for 0.5 m/s | — | ~39 fps |

**The key mental model (important):** the bug did NOT make the robot move 1.67× faster. It made the *reference* step 1.67× too fast — past sonic's trackable bandwidth. Sonic is a finite-bandwidth tracking policy (PD-driven, trained on a specific reference cadence); when the reference joint targets sweep faster than anything it saw in training, it **lags and the intended motion is smeared / attenuated / lost**, not sped up. Feet: the reference foot placement moves faster than sonic can plant→lift → the foot slides (**slippage**). Run mode: the reference legs cycle fast but sonic can't drive the body that fast → **"runner pose, barely moves"** — the motion was *lost*, not accelerated. Fixing the rate put the reference back inside sonic's envelope, so it can actually reproduce it — which is why the walk immediately looked clean.

The offline eval avoided all this because `eval_x2_mujoco.py` samples the motion by **sim-time → frame at the real `motion_fps` (30)** (line ~491), i.e. it *accidentally did the resampling the live path skipped.*

## What the G1 stock stack does (the spec we were missing)

From `docs/source/references/planner_onnx.md` §"Output Resampling (30 Hz → 50 Hz)" and §"Animation Blending" (the authoritative reference implementation):

1. **Output resampling 30 Hz → 50 Hz** — for each 50 Hz frame, compute the **fractional 30 Hz frame index**; **lerp** joint/body positions, **slerp** body quaternions between the two nearest 30 Hz frames; joint velocities = finite-difference of the resampled positions × 50. Planner runs on a dedicated 10 Hz thread; control loop 50 Hz.
2. **8-frame cross-fade blend** — when a new planner output arrives while the previous is still playing, blend old→new over an 8-frame window (linear `w_new` 0→1; slerp quats) so successive predictions have no visible seam.
3. (G1 also resamples the **context** the other way — 4 frames at 30 Hz intervals lerp/slerp'd from the 50 Hz stream. X2 didn't need this: its ring already holds native 30 fps content and the reseed samples at 1/30 spacing.)

**X2 had neither** — it advanced +1 frame/tick (no resample) and bolted on a crude reactive `ref-smoother` (300 ms halfcos on >0.05 rad jumps) that masked seam jumps by **shifting foot timing** (itself a slippage source).

## The fix

Implemented the G1 handoff in the **shared** `NeuralPlannerCore` so both runtimes inherit it:

- **`motionbricks/.../inference/neural_planner.py`** — float read cursor `_read_pos` (native 30 fps units) advancing `model_fps/OUTPUT_FPS = 0.6` per 50 Hz tick; `_frame_at()` (lerp trans[0:3]+dof[7:38], slerp root-quat[3:7]); `get_next_frame_resampled(output_fps)`, `peek_output_frame()` (futures on the same timeline); resample-aware `should_replan()`; 8-frame cross-fade (`_arm_output_blend`, `w_new` 0.125→1.0 over 8 ticks) armed on each replan before the buffer swap. All gated on `_resample_active` so the legacy integer path is byte-identical (offline tooling unaffected).
- **`gear_sonic/scripts/x2_kplanner.py`** (torch live stack) — PLAYING branch uses `get_next_frame_resampled` + `peek_output_frame`; also fixes the future-window spacing to be time-correct (`step_ticks=5` × 0.6 = +0.1 s real, was +0.167 s). Ref-smoother default `halfcos → off`.
- **`gear_sonic/scripts/pc2_kplanner_onnx.py`** (robot ONNX runtime) — numpy port of the same resample/blend (`_slerp_wxyz_np`, `MODEL_FPS=30`); torch/numpy resample parity < 6e-8. Ref-smoother default → off.

## Validation

Standalone (`scratchpad/validate_resample.py`), FT trio 250k, constant 0.5 m/s, 6 s, open-loop:

| metric | OLD (+1/tick) | NEW (resample+blend) | note |
|---|---|---|---|
| forward speed (cmd 0.5) | 0.994 m/s | 0.596 m/s | **OLD/NEW = 1.668 = the 30/50 fix, exactly** |
| replan-seam joint jump | 0.282 rad | 0.154 rad | 8-frame blend halves it |
| leg cadence | 1.333 Hz | 0.667 Hz | 2× de-inflated to natural pace |

- Torch vs ONNX resample parity: max diff < 6e-8.
- Offline replay path unchanged (RMS 0.149) — integer path byte-identical.
- Ref-smoother + pose-feedback test suites pass.
- **Hardware-in-sim visual A/B:** live gamepad walk went from slipping/hurried to *smooth, natural cadence, no slippage* — matching the offline quality.

(The NEW open-loop speed 0.596 is ~19% over 0.5; the deploy stack's closed-loop pose reseed pulls it back toward the command. The load-bearing results are the 1.668 ratio and 2× cadence de-inflation.)

## Takeaways

- **When X2 deploy behavior diverges from expectation, diff against the G1 stock stack first.** `docs/source/references/planner_onnx.md` documents the reference handoff; the X2 port silently skipped the resample+blend.
- **Isolate stages before blaming models.** Planner-alone, sonic-on-raw-output, and live-loop are three separately-viewable stages; the fault was in none of the models but in the glue between them.
- Residual, separately-scoped: the **root model under-translates** (0.388 m/s for a 0.5 command) — a data/supervision gap, not a handoff bug. And the **g1teleop walk recordings have near-frozen arms** (raw human `pose_aa` R-shoulder std ≈ 0.000) — the kplanner actually *adds* swing on top.

## Related

- Root cause + fix status memory: `kplanner-handoff-fps-fix`
- G1 reference: `docs/source/references/planner_onnx.md` (§ Output Resampling, § Animation Blending)
- PC2 untether runbook: `x2_upgraded_demo/pc2_untethered_stack.md`
