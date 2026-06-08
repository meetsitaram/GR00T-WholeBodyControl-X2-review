# 2026-06-07 — Calibration palm-direction lock + multi-pose wrist-offset (v1)

> **Session focus.** Operator wrist orientation tracking was visibly off
> on the kinematic teleop (robot wrist bent toward pinky / forward at
> "natural" grip). Root cause: `wrist_alignment_quat` is fit from
> arms-down ONLY and assumed constant across all poses, while the Quest
> 3 controller pivots in the operator's grip as wrist orientation
> changes. Calibration instructions also lacked palm-direction lock, so
> the alignment quat varied between sessions even with no other change.
>
> This milestone documents the v1 fix (palm-direction lock + multi-pose
> least-squares wrist offset derived from the calibration's own
> measurements). It is a 60–70% solution; ~8–16° of residual wrist
> deflection per pose remains. The "what to iterate on" section below
> enumerates the cleaner fixes we deferred so we can pick them up
> later.

---

## TL;DR

| Symptom (before) | Cause | Fix (v1) |
|---|---|---|
| Robot wrist visibly bent toward pinky (yaw) and forward (pitch saturated at +32°) when operator holds wrists straight along forearm. Worse in arms-forward than arms-down. | `wrist_alignment_quat` in the calibration YAML is fit from **arms-down only** (`operator_calibration.py:869-877`). The Quest 3 controller's body axis pivots ~10-20° relative to the operator's wrist depending on the wrist's orientation in space, so the single alignment that's correct at arms-down asks the IK for **up to 120° of impossible rotation at arms-forward** (joint range is ±32° pitch). | (a) Lock palm direction in calibration prompts so the arms-down reference is reproducible. (b) Refit a single wrist_alignment as a least-squares quaternion average over `arms_down + t_pose + arms_forward` (namaste is biomechanically too distorted to be useful). (c) Expose the resulting delta as a constant operator-frame RPY offset via two new CLI args. |
| Calibration's spoken/printed pose instructions never specified palm orientation. Result: even back-to-back calibrations on the same operator captured ~5–9° wrist-orientation drift, and across separate sessions the drift hit ~25–40°. | `CALIBRATION_POSE_INSTRUCTIONS` and the TTS `PROMPT_TEXTS` only described *arm* posture (down/T/forward/namaste), not palm direction. | Added "palms face inward toward thighs / palms face down / palms face each other" to every pose instruction + spoken prompt. Nuked the cached `show_*.mp3` / `recapture_*.mp3` so the next calibration regenerates with the new text. |
| Quat convention error in our diagnostic scripts inflated the apparent offset to ~180°. | The NPZ stores wrist quats as **wxyz** (scalar-first, see `quest3_reader.py:127` `as_quat(scalar_first=True)` and docstring `[x,y,z,qw,qx,qy,qz]`), not xyzw as an earlier conversation summary stated. Components were swapped and produced phantom 180° rotations about +X. | Diagnostic scripts and analyses in this milestone use wxyz directly. |

---

## What landed

### Calibration prompts (palm-direction lock)

- `gear_sonic/utils/teleop/operator_calibration.py` —
  `CALIBRATION_POSE_INSTRUCTIONS` now specifies palm direction per pose
  and adds the line *"Hold the controllers the same way you will
  during teleop."*
- `gear_sonic/utils/teleop/vr/quest3_audio_prompts.py` — same wording
  in the spoken `show_*` and `recapture_*` TTS prompts.
- Cached MP3s nuked (`gear_sonic/utils/teleop/vr/quest3_webxr_app/audio/show_*.mp3` and `recapture_*.mp3`) so the next calibration regenerates fresh audio from the updated text.

Pose-direction table that the new prompts encode:

| Pose | Palm direction |
|---|---|
| `arms_down` | palms inward toward thighs |
| `t_pose` | palms down toward the floor |
| `arms_forward` | palms facing each other |
| `namaste` | palms together (self-defined) |

### Persisted in the calibration YAML

- `gear_sonic/utils/teleop/operator_calibration.py` — `ArmFit` gained
  an optional `op_quat_offset_rpy_deg` field (defaults to zeros, so v0
  / v1 YAMLs still round-trip unchanged). When non-zero it serialises
  as `fit.<side>.op_quat_offset_rpy_deg: [roll, pitch, yaw]` and is
  loaded back into the dataclass on `OperatorCalibration.load_yaml`.
- `gear_sonic/utils/teleop/vr_arm_teleop_v2.py` — three-tier
  resolution at `VRArmTeleopCalibrated.__init__`:
  1. Explicit constructor kwarg (CLI overrides win).
  2. `calibration.fit[side].op_quat_offset_rpy_deg` (YAML-default).
  3. Identity (no offset).
  Construction logs which tier was used, e.g. `wrist op-quat offsets
  active: left_rpy_deg=(-5.7, 1.8, -6.0) (calibration-yaml); ...`.
- **Net effect**: every launcher that builds a `VRArmTeleopCalibrated`
  or a `Retargeter` (kinematic teleop, dataset recording via
  `quest3_manager_x2.py`, SONIC bridge) now picks up the same per-
  operator wrist offset automatically — no per-script CLI plumbing.

### Kinematic teleop — operator-frame wrist offset (CLI)

- `gear_sonic/scripts/teleop_x2_kinematic.py` — args
  `--left-wrist-op-quat-offset-rpy-deg` and
  `--right-wrist-op-quat-offset-rpy-deg`, both 3-float intrinsic XYZ
  Tait-Bryan degrees, post-multiplied on the operator's head-yaw-frame
  wrist quat in the operator wrist local frame BEFORE the
  calibration's `wrist_alignment` runs. **Now optional** — when
  omitted, the value from the calibration YAML is used; pass a tuple
  here (including `0 0 0`) to override. Reuses the existing kwargs of
  `VRArmTeleopCalibrated`.
- `gear_sonic/scripts/quest3_manager_x2.py` — same `--left-wrist-
  offset-rpy-deg` / `--right-wrist-offset-rpy-deg` flags also default
  to `None` (was `(0,0,0)`, which would have suppressed the YAML
  value). The Quest 3 manager config dataclass field is now
  `Optional[tuple[...]]`.

### Recommended offset values (this operator, calibration as of 2026-06-07)

Derived as `inv(wa_arms_down) * R_LS_avg(arms_down, t_pose, arms_forward)`,
where `R_LS_avg` is a Markley quaternion least-squares average of the
per-pose ideal alignments `R_p = rh_p · inv(op_p)`.

| Side | Roll | Pitch | Yaw | Mag |
|---|---:|---:|---:|---:|
| LEFT | -5.7° | +1.8° | -6.0° | 8.5° |
| RIGHT | +0.3° | +4.4° | +10.4° | 11.3° |

These values are now baked into `data/operator_calibrations/default.yaml`
under `fit.left.op_quat_offset_rpy_deg` /
`fit.right.op_quat_offset_rpy_deg`. To use them just run any launcher
with the default calibration — no flags needed:

```bash
.venv/bin/python -m gear_sonic.scripts.teleop_x2_kinematic \
    --output-dir /tmp/ik_debug \
    --task "kinematic teleop" \
    --rate 50
```

Look for this line on startup as confirmation the offsets are active:

```
[VRArmTeleopCalibrated] wrist op-quat offsets active: \
    left_rpy_deg=(-5.7, 1.8, -6.0) (calibration-yaml); \
    right_rpy_deg=(0.3, 4.4, 10.4) (calibration-yaml).
```

To override the YAML value for one-off experiments (e.g. tuning a new
offset), pass the flag explicitly:

```bash
.venv/bin/python -m gear_sonic.scripts.teleop_x2_kinematic \
    --output-dir /tmp/ik_debug \
    --task "kinematic teleop" \
    --rate 50 \
    --left-wrist-op-quat-offset-rpy-deg  -5.7 +1.8 -6.0 \
    --right-wrist-op-quat-offset-rpy-deg  +0.3 +4.4 +10.4
```

Passing `0 0 0` explicitly suppresses the YAML value (useful when
running the parity test or A/Bing against the old behaviour).

### Per-pose residuals after the LS-average fit (3 poses)

| Pose | LEFT residual | RIGHT residual |
|---|---:|---:|
| arms_down | 8.5° | 11.3° |
| t_pose | 11.2° | 15.6° |
| arms_forward | 9.3° | 11.5° |

For reference, the arms-down-only `wrist_alignment` (the production
fit) had 0° residual at arms-down but **120° / 109° residual at
arms-forward (L / R)** — totally unachievable by the joints. So the LS
fit trades a small (~10°) error at arms-down for usable behaviour
across the manipulation pose space.

---

## How we got there (diagnostic notes worth keeping)

These are the data points that drove the v1 design. Useful when we
re-open this work.

1. **`wrist_alignment_quat` is single-pose by design.** See
   `gear_sonic/utils/teleop/operator_calibration.py:869-877`. The
   docstring on `ArmFit.wrist_alignment_quat` (line 441) even calls
   this out: "derived from the arms-down pose."
2. **Cross-pose ideal-alignment drift is large.** Computing
   `R_p = rh_p · inv(op_p)` per pose from the same calibration:
   - L pairwise drift: arms_down↔t_pose 17.5°, arms_down↔arms_forward 13.9°, *arms_down↔namaste 56.0°*
   - R pairwise drift: 24.6°, 16.8°, *72.2°*
   Namaste is the outlier — palms-touching squeezes the wrist into a
   non-representative grip — and is excluded from the LS fit. The
   three "manipulation" poses are consistent enough for a single
   compromise alignment.
3. **Recording-vs-calibration drift on the SAME operator.** Even
   immediately after a fresh palm-direction-locked calibration, the
   recording's arms-down wrist quat differs from the calibration's
   stored one by ~5-9° per axis. That ~10° on the operator side gets
   multiplied through wrist_alignment into a ~20° IK target deviation,
   which is enough to saturate the ±32° wrist_pitch joint at
   arms-down.
4. **The IK *target* delta from robot home is what saturates the
   joints, not the wrist_alignment magnitude alone.** Reconstructing
   the live IK target from the offset-test NPZ:
   - arms-down: target is 19-21° off robot home (within joint range,
     but combined with kinematics-chain coupling forces wrist_pitch to
     +32° saturation)
   - arms-forward: target is 109-120° off robot home (no chance)

---

## Known limitations / what to revisit next time

> Open this section first when we come back to wrist offsets. Each
> item is a concrete iteration we deliberately deferred.

1. **~~The offsets are not persisted yet.~~ ✅ DONE (2026-06-07).**
   Persisted as `fit.<side>.op_quat_offset_rpy_deg` on `ArmFit` and
   resolved at `VRArmTeleopCalibrated.__init__` with the three-tier
   order (CLI > YAML > identity). Confirmed picked up by
   `teleop_x2_kinematic` and `quest3_manager_x2` (which also moved
   its dataclass field from `(0,0,0)` default to `None`, so passing
   `(0,0,0)` no longer accidentally suppresses the YAML value).
2. **Single constant offset is structurally limited to ~10° residual
   per pose.** The operator's controller-grip tilt is not constant
   across arm poses; the controller body pivots a few degrees as the
   wrist rotates in space. Two cleaner architectures:
   * **Per-pose interpolation.** Store per-pose `wrist_alignment_quat`
     in calibration, then at runtime interpolate via barycentric
     weights based on the current wrist-position barycentric weights
     in the 4-pose simplex (or use the closest 3 of the 4). Should cut
     residual to <5°.
   * **Pose-conditioned residual MLP.** Train a tiny MLP on the
     calibration measurements that maps (arm_pose → wrist_alignment_quat).
     Overkill for a one-operator setup but cheap.
3. **`wrist_alignment_quat` fit excludes namaste by design — but the
   prompt still includes namaste.** We could just drop namaste from
   `CALIBRATION_POSE_IDS` for the alignment fit (keep capturing it for
   future use), or keep it in the schema but explicitly weight it down
   in the LS step.
4. **No automated way to recompute the offset from a calibration
   YAML.** The numbers in this milestone were computed by an ad-hoc
   script (`docs/source/user_guide/milestones/2026-06-07_wrist_offset_v1.md`
   reproduces the math). **Action**: ship
   `gear_sonic/scripts/compute_wrist_offset_from_calibration.py` that
   loads a calibration YAML, runs the LS fit over the 3 manipulation
   poses, and prints the suggested offset (with an optional
   `--write-yaml` to persist back into the same YAML).
5. **Quat convention is fragile.** The NPZ stores wxyz, the IK takes
   xyzw at one boundary and wxyz at another, the calibration YAML
   stores wxyz, scipy uses xyzw, and a previous summary asserted xyzw
   for the NPZ. We burned several debugging hours on this. **Action**:
   add a `WXYZ` / `XYZW` newtype / NewType annotation at every
   boundary, and a runtime sanity assertion (`abs(w) < 0.99 and
   abs(x) > 0.95` ⇒ probably swapped).
6. **The user demonstrated that the IK target IS reachable** — by
   manually twisting their wrist toward the thumb, they get a
   straight-looking robot wrist. So the v1 LS-fit offset is a coarse
   approximation of *exactly* this manual twist. A bespoke recording
   ("hold natural for 8s, then compensate for 8s, save") would give a
   per-pose ground truth offset to validate the LS-fit against. We
   skipped this because the operator was understandably tired of
   recording, but it's the cleanest validation if we re-iterate.

---

## How to verify

1. Capture a kinematic-viewer screenshot at three states:
   * **Baseline** (no offset CLI args) — robot wrist visibly bent toward pinky and/or forward.
   * **v1 offset applied** (CLI args from "Recommended offset values" above) — wrist deflection noticeably reduced; should look usable in arms-down and arms-forward both.
   * (Optional) compute live IK wrist joints `q[4:7]` from the
     resulting recording's `ik_left_q_rad[:, 4:7]` and confirm mean
     |yaw| and |roll| dropped vs baseline.
2. Cross-pose sanity: do an arms-down → arms-forward → T-pose sweep
   and confirm the wrist doesn't look dramatically more bent in any
   of the three.

---

## File index

Source changes (operator calibration + audio):

- `gear_sonic/utils/teleop/operator_calibration.py` — `CALIBRATION_POSE_INSTRUCTIONS` updated.
- `gear_sonic/utils/teleop/vr/quest3_audio_prompts.py` — `PROMPT_TEXTS["show_*"]` and `PROMPT_TEXTS["recapture_*"]` updated.
- `gear_sonic/utils/teleop/vr/quest3_webxr_app/audio/show_*.mp3`, `recapture_*.mp3` — deleted, regenerate on next calibration.

Source changes (CLI):

- `gear_sonic/scripts/teleop_x2_kinematic.py` — added `--left-wrist-op-quat-offset-rpy-deg` and `--right-wrist-op-quat-offset-rpy-deg`, wired into `VRArmTeleopCalibrated.__init__`.

Calibration YAML:

- `data/operator_calibrations/default.yaml` (fresh palm-direction-locked calibration, residual ≤ 0.16 m).
- `data/operator_calibrations/default.before_palm_direction_20260608_012140Z.yaml.bak` (backup pre-rewrite).
- `data/operator_calibrations/default.before_20260607_224557Z_recording.yaml.bak` (earlier backup).
