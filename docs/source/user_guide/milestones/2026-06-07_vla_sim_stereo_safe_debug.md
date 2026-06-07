# 2026-06-07 — VLA sim-first debug path with stereo cameras (M5 v1)

> **Session focus.** First powered VLA runs against the
> `x2_grab_a_drink_n17_30k_v1/checkpoint-25000` real-robot fine-tune
> exhibited severe vibration and, after the initial proprio + ramp +
> LPF mitigations were merged, joint deviations of up to ~2.77 rad on
> the second chunk handoff. The decision was made to stop debugging on
> hardware and extend the existing sim path
> (`run_live_vla_demo.sh`) to support the **omnihand-stereo** modality
> used by the real-robot checkpoints, so the same closed loop can be
> diagnosed safely in the MuJoCo viewer.

---

## TL;DR

| Aspect | Status |
|---|---|
| Real-robot VLA bring-up (with 990-D proprio, ramp-in, LPF, chrony) | ⚠️ unsafe; joint deviation up to ~2.77 rad on chunk 2 |
| Sim-side parity for the multi-key modality | ✅ this milestone |
| `_GhostCameraProvider` (multi-key MuJoCo renderer) | ✅ new in `live_vla_publish_motion_token.py` |
| `run_live_vla_demo.sh` default modality flipped to `omnihand_stereo` | ✅ matches the real-robot checkpoint |
| `--motion-token-decoder` + `--vla-ramp-in-ticks` + `--vla-target-lpf-hz` wired through the demo launcher | ✅ same flags as the real-robot launcher |
| Architecture doc updated with the sim-parity section + ghost camera mapping table | ✅ `x2_sonic_runtime_architecture.md §7` |
| End-to-end MuJoCo viewer rollout with the trained checkpoint | ⏭️ next session — operator step |

---

## Why this milestone exists

The earlier sim path (M5 v0, 2026-05-08) only rendered a single
`ego_view` camera because the synthetic-smoketest fine-tune declared
`video.modality_keys=["ego_view"]`. The real-robot fine-tune
(`x2_grab_a_drink_n17_30k_v1`) was trained with
`x2_modality_config_omnihand_stereo.py`, which declares two video keys
(`stereo_left + stereo_right`). Loading the real-robot checkpoint
against the legacy sim path silently failed at observation build time —
the policy got no images on those keys — so sim simply wasn't a
debugging option for the checkpoint that was actually being deployed.

The patch is intentionally narrow:

1. **No MJCF edits.** The X2 MJCF already has a `stereo_head_front`
   mount, but with a single optical centre. We render that mount once
   per tick and alias the resulting frame under both `stereo_left` and
   `stereo_right`. The policy sees identical L and R — "degenerate
   stereo" — which is enough to validate the pipeline (wire, proprio,
   ramp, LPF, decoder, SONIC body motion) but not to validate the
   model's depth reasoning. True stereo (separate L/R MJCF cameras with
   a ~5 cm baseline) is the natural follow-up.
2. **No bridge surface changes beyond the camera provider.** The
   inference worker, observation builder, modality config loader, and
   wire shaping all stay byte-identical between sim and real-robot.
   Only the camera source swaps.

---

## What changed

### New: `_GhostCameraProvider`

`gear_sonic/scripts/live_vla_publish_motion_token.py` gains a sibling
of `_RealCameraProvider`. It looks at the modality config's
`video.modality_keys`, maps each key to an MJCF camera through a
small `MODALITY_TO_MJ_CAMERA` table, builds one `MujocoFrameRenderer`
per unique MJCF camera (inside the inference thread because EGL is
thread-local), and returns the per-tick frame dict the existing
`_build_observation` already expects.

Mapping table:

| Modality key | MJCF camera | Notes |
|---|---|---|
| `ego_view`, `rgbd`, `rgbd_head_front`, `head_front` | `rgbd_head_front` | Egocentric RGB-D mount |
| `stereo_left`, `stereo_right`, `stereo`, `stereo_head_front` | `stereo_head_front` | Single optical centre — L/R alias |
| `rgb_head_center` | `rgb_head_center` | Direct passthrough |
| `rgb_head_rear` | `rgb_head_rear` | Direct passthrough |

The `_inference_worker` now takes a `ghost_provider_factory` instead
of the old single-`renderer_factory` callable. The legacy single-key
(`ego_view`) sim modality still works — the provider just builds one
renderer for `rgbd_head_front`.

### Modified: `run_live_vla_demo.sh`

- Default `MODALITY` now `x2_modality_config_omnihand_stereo.py`
  (was implicit, single-key).
- New `MOTION_TOKEN_DECODER` env var (`.pt` path). Preflight hard-fails
  if it's unset or missing — without it the body never moves under VLA
  authority.
- New `VLA_RAMP_IN_TICKS` (default 25 = 0.5 s) and
  `VLA_TARGET_LPF_HZ` (default 8 Hz) wired through to the bridge;
  matches the real-robot launcher.
- `TQDM_DISABLE=1` in the bridge's environment so the
  "Loading weights: 100%" carriage-return bars stop spamming
  `bridge.log` (same fix as `run_x2_vla_runtime.sh`).
- Preflight now also probes for `msgpack_numpy` import.

### Modified: `docs/source/references/x2_sonic_runtime_architecture.md`

New section 7 "Sim-side parity — `run_live_vla_demo.sh` for safe VLA
debugging" with the data-flow diagram, parity table, ghost camera
mapping table, operator commands, and the things-to-watch checklist
for `bridge.log`. Section 8 renumbered from 7 (Pointers).

### Modified: `pick_place_commands.md`

New "Sim-first VLA debugging" section with the one-liner for running
the real-robot checkpoint in sim and the parity matrix.

---

## What's left for the next session

1. **Run the real-robot checkpoint in sim.** Operator step:

   ```sh
   ./gear_sonic/scripts/run_x2_vla_runtime.sh \
       --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
       --motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
       --prompt "grab the can from the table"
   ```

   Watch the MuJoCo viewer for the chunk boundary the powered run
   diverged on. Compare `raw_Δ` (policy intent) and `wire_Δ` (post
   ramp + LPF) in `bridge.log`. Expect `raw_Δ` ≪ 1 rad once the policy
   has plausible proprio + a plausible image; if it still spikes to
   2 rad+, the issue is the checkpoint, not the wire pipeline.
2. **True stereo MJCF** if visual reasoning quality matters: add two
   cameras to `HEAD_CAMERAS` with a ~5 cm baseline, extend
   `_GhostCameraProvider.MODALITY_TO_MJ_CAMERA` to map them, and the
   provider will build both renderers automatically.
3. **Compare sim vs real-robot trajectories** for the same chunk dump
   to isolate whether the visual OOD is the dominant failure mode.

---

## References

- M5 v0 milestone (single-camera sim path):
  [`2026-05-08_live_vla_sonic_sim_v0`](2026-05-08_live_vla_sonic_sim_v0.md)
- Cross-mode architecture (teleop/record/VLA shared backbone):
  [`x2_sonic_runtime_architecture`](../../references/x2_sonic_runtime_architecture.md)
- Bridge-side decoder math + proprio assembly:
  [`x2_vla_motion_token_decoder`](../../references/x2_vla_motion_token_decoder.md)
- VLA real-robot runbook (the one that was unsafe — fix in sim first):
  [`x2_vla_runtime`](../../tutorials/x2_vla_runtime.md)
