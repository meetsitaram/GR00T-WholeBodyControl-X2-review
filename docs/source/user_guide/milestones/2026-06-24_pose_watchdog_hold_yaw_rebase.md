# 2026-06-24 — PC2 pose watchdog HOLD state yaw-rebases cached frames

> **Status: ✅ Code landed, unit tests green.** Real-robot validation
> pending (operator request was to fix the "robot snaps back to its
> previous orientation when I kill the PKL-direct stack" symptom).
>
> **Session focus.** An operator running
> `gear_sonic/scripts/run_x2_pkl_direct_stack.sh --pc2-host …`
> reported two related symptoms during PKL playback on the real
> robot:
>
> > "the pkl direct stack doesn't want to lose its orientation even
> > during stabilizing moves and when there is no pkl file playing"
>
> and on follow-up:
>
> > "just by running ./gear_sonic/scripts/run_x2_pkl_direct_stack.sh
> > --pc2-host 192.168.86.32, it locks. also after i end that, it
> > tries going back to previous locked position abruptly too"
>
> The post-Ctrl-C symptom turned out to be a known foot-gun in the
> PC2 watchdog's `HOLD` state: it re-publishes the cached upstream
> frame byte-for-byte, including a `root_quat_xyzw` field whose yaw
> is now stale relative to the robot's actual heading. The SONIC
> policy responds to the stale absolute-yaw reference by twisting
> the body back to the cached heading for up to `--hold-last-secs`
> (default **5 s**) before `BLEND`/`IDLE_CLIP` take over and
> re-engage their existing yaw-rebase paths.
>
> **Subsystem reference.** This continues the
> [2026-06-23 VLA bridge yaw hold-last-good](2026-06-23_vla_bridge_yaw_hold_last_good.md)
> story by closing the symmetric gap on the PC2 side. Where the
> bridge fix made the *laptop* honest about the live yaw across
> `x2_debug` stalls, this fix makes the *watchdog* honest about the
> live yaw across upstream-pose stalls. For the full architectural
> picture see
> [`x2_heading_stability_and_yaw_rebase`](../../references/x2_heading_stability_and_yaw_rebase.md).

---

## TL;DR

| Symptom (before) | Cause | Fix |
|---|---|---|
| Operator runs `run_x2_pkl_direct_stack.sh --pc2-host …`, plays a clip, presses Ctrl-C, then tries to push the robot to a new heading by hand. For up to 5 s after Ctrl-C the body fights back to whatever yaw the laptop was publishing when killed, then abruptly drifts toward the baked stand pose. | The watchdog's `STATE_HOLD` branch was `pub.send(last_upstream_msg, …)` — a pure byte-for-byte re-publish of the cached frame. The cached `root_quat_xyzw` (and the 9 future-window slots if present) describe a stale absolute yaw the moment the operator tries to rotate the body. SONIC reads the stale reference and provides restoring torque for the full `--hold-last-secs` window. | New `rebase_hold_msg` helper in `gear_sonic/utils/pose_pipeline/fallback.py`: surgically splice `R_z(measured_yaw)` (from the live `x2_debug` cache) into the cached frame's `root_quat_xyzw` and `root_quat_xyzw_future` fields via `rebuild_msg_with_field_overrides`. Joint targets are left bit-identical, so the body pose still freezes exactly where upstream left it — only the heading reference tracks the robot's actual yaw. Falls back to verbatim re-publish if `x2_debug` is stale or the splice fails (legacy v4 header missing the field). |
| `[pose_watchdog]` status lines silently treated all `HOLD` ticks as equivalent. No way to tell whether yaw-rebase was actually engaged during a given HOLD cycle. | Counter only tracked `hold_frames` (total). | Add `hold_frames_with_rebase` counter, emit a separate `hold_rebased=N` field in the periodic status line and in the final shutdown summary. One-shot `ACTIVE` / `FALLBACK` / `splice FAILED` log lines mirror the recorder's `idle yaw-rebase: ACTIVE / fallback` gates so a 50 Hz HOLD loop emits at most three lines per state-edge, not 150/s. |

---

## Why the existing yaw-rebase paths didn't already cover this

The watchdog already yaw-rebases two of its three idle-fallback
branches. From `gear_sonic_deploy/scripts/x2_pose_watchdog.py`:

* **`STATE_BLEND`** — calls `build_idle_frame_msg(..., yaw_rebase_rad=yaw_rebase)`,
  where `yaw_rebase` is the last measured pelvis yaw from `x2_debug`.
  The idle clip's baked `R_z(0)` quats are pre-rotated by
  `R_z(measured_yaw)` before lerping with the cached upstream futures.
* **`STATE_IDLE_CLIP`** / **`STATE_COLD_IDLE`** — same call, same
  rebase semantics. The baked clip never reaches the wire with its
  authored identity yaw.

But **`STATE_HOLD`** was special-cased: it bypassed
`build_idle_frame_msg` entirely and re-published the raw cached bytes
via `pub.send(last_upstream_msg, …)`. The design intent was correct
for its original purpose — keep the deploy seeing a stable
`joint_pos -> jvel = 0` reference during sub-second wifi blips so the
arms don't drift. But it had a hidden assumption: that the cached
quat is *also* valid for the duration of the HOLD window. That
assumption breaks the moment the operator either (a) lets `x2_debug`
diverge from the cached value (by pushing the robot, or by waiting
long enough for natural pose drift), or (b) intentionally kills the
upstream stack and tries to interact with the body.

The fix lifts the same yaw-rebase pattern into HOLD, but with a
narrower scope than `BLEND`/`IDLE_CLIP`: we don't touch
`joint_pos_mj` (HOLD's contract is "freeze the body pose"); we only
rewrite the root quat fields.

---

## What landed

### `gear_sonic/utils/pose_pipeline/fallback.py`

1. **`_yaw_to_quat_xyzw(yaw_rad: float) -> np.ndarray`** — centralised
   half-angle / dtype conversion for the pure-yaw quat. Matches
   `rebase_quats_xyzw_by_yaw`'s convention so the HOLD path and any
   future call sites agree on float precision and array layout.
2. **`rebase_hold_msg(last_upstream_msg, topic, yaw_rad) -> bytes | None`**
   — the new public helper. Tries a two-field splice first
   (`root_quat_xyzw` + `root_quat_xyzw_future`); falls back to a
   one-field splice (`root_quat_xyzw` only) for legacy v4 producers
   that omit the future window. Returns `None` if both attempts fail,
   so the watchdog can drop back to verbatim re-publish without
   sending a malformed frame.
3. **All 9 future slots get the same `R_z(yaw_rad)` quat.** HOLD
   semantically means "no anticipatory motion" — flattening the
   future window is intentional. The cached future may encode an
   authored yaw delta from a locomotion clip's final frame; honouring
   those deltas with upstream silent would re-engage the policy's
   planning horizon to steer toward a missing clip's authored
   trajectory, which is worse than freezing.

### `gear_sonic_deploy/scripts/x2_pose_watchdog.py`

1. **`STATE_HOLD` branch rewritten.** When `yaw_track_enabled` is true
   and `last_measured_yaw_s` is fresher than `--x2-debug-max-age-s`
   (default 0.5 s), call `rebase_hold_msg` on the cached frame; send
   the rebased bytes. On splice failure or stale `x2_debug`, fall
   back to `pub.send(last_upstream_msg, …)` (today's verbatim path).
2. **`hold_frames_with_rebase` counter + status-line plumbing.**
   Increments only when the rebased message is what actually went
   out. Surfaced as `hold_rebased=N` in the periodic status print
   and in the final shutdown summary alongside the existing
   `idle_rebased=N`.
3. **Three one-shot log lines** (mirror `_idle_yaw_rebase_logged_*`):
   * `HOLD yaw-rebase: ACTIVE -- … R_z(measured_yaw=…)` on the first
     successful splice of a HOLD cycle.
   * `HOLD yaw-rebase: x2_debug went stale; falling back to verbatim
     cached-frame re-publish` on the live → stale edge.
   * `HOLD yaw-rebase: splice FAILED …` once per session if the
     cached frame's header doesn't carry a `root_quat_xyzw` field.
4. **Startup banner line.** `[pose_watchdog] HOLD yaw-rebase:
   ENABLED / DISABLED` so operators see the new feature in the
   first 8 lines of the watchdog log.

### Tests — `tests/test_x2_pose_watchdog_fallback_ladder.py`

Renamed the existing `test_hold_path_is_byte_identical_republish`
to `test_hold_path_verbatim_fallback_is_byte_identical` (now scoped
to the verbatim *fallback* path, since the active HOLD path is
exercised by the new tests). Added 5 new tests:

* `test_rebase_hold_msg_v4_message_rebases_current_quat` — recorder
  `_publish_idle` v4 frames (no future window) still get the current
  quat rebased; `joint_pos_mj` is bit-identical.
* `test_rebase_hold_msg_v5_message_rebases_current_and_future` —
  cached future window with authored yaw deltas gets flattened to a
  single `R_z(measured_yaw)` quat across all 9 slots; jpos
  untouched.
* `test_rebase_hold_msg_wrong_topic_returns_none` — topic mismatch
  fails safely.
* `test_rebase_hold_msg_zero_yaw_produces_identity_quat` — guards
  against an off-by-one in the half-angle conversion.
* `test_rebase_hold_msg_preserves_motion_token_and_hand_joints` —
  only root-quat bytes change; motion_token, hand joints,
  frame_index, future jpos all bit-identical; output length equals
  input length (no header rewrite).

All 25 tests in `test_x2_pose_watchdog_fallback_ladder.py` pass
(20 pre-existing + 5 new). The broader pose-pipeline regression
sweep (93 tests across `test_pose_pipeline_*`, `test_x2_pose_*`)
also passes.

---

## What is NOT touched

* **`--hold-last-secs` default (5.0 s)** — left alone. With HOLD now
  yaw-rebased, the 5 s window is back to its original safe purpose
  (soak up wifi blips without forcing the arms to glide). Shrinking
  it would only matter if HOLD was still doing something wrong;
  it's now strictly better than the BLEND lerp it would prematurely
  hand off to.
* **Recorder-side idle publishing.** The recorder's `_publish_idle`
  already rebases its own `root_quat_xyzw` per
  `x2_dataset_recorder.py:3367-3450`; the only gap was in the
  watchdog's HOLD path that ate stale cached frames. This fix is
  on the consumer side of the wire so it protects both
  recorder-stack shutdowns AND any other upstream that silently
  goes stale (mux death, mock_vla crash, VLA bridge stall during
  GPU OOM).
* **`STATE_HOLD` semantics for `joint_pos_mj`.** Still bit-identical
  to upstream's last frame. The body freezes; only heading tracks.
* **The C++ deploy.** No change. The deploy sees the same wire
  field layout it always did; only the quat bytes differ on a
  per-tick basis during HOLD-with-rebase.
* **`--idle-mode hold-last`** — same `STATE_HOLD` branch, so it
  inherits the rebase for free. Operators using indefinite-HOLD
  mode (recovery scripts, manual takeover gating) get the same
  benefit.

---

## Open follow-ups

1. **Real-robot soak test.** Operator launches the stack, plays a
   clip, kills the stack, attempts to rotate the body during the
   first 5 s. Symptom should be: body holds joint pose but yields
   freely to operator-induced rotation; periodic status line shows
   `hold_rebased=N` ticking with `hold=N` (1:1).
2. **In-stack "stiffness" complaint** ("just by running it locks").
   The recorder's idle publish DOES rebase per tick, but with a
   ~50–100 ms wifi RTT lag a fast operator rotation feels springy.
   That's a separate (smaller) symptom; not addressed here. Options
   for a follow-up:
   * Predict yaw forward by `measured_yaw_rate * lag_compensation_s`.
   * Add a recorder-side `--free-yaw-idle` mode that uses a slower
     yaw-update cadence (or a wider deadzone) so small operator
     nudges leak through.
   * Document `--no-idle-publish` (already wired through
     `run_x2_pkl_direct_stack.sh`) as a one-line workaround: the
     recorder stays silent during idle and the watchdog's
     `IDLE_CLIP` (also yaw-rebased) takes over. Acceptably
     low-fidelity for preview-only sessions.
3. **Update `references/x2_heading_stability_and_yaw_rebase.md`.**
   The existing reference notes that the watchdog has yaw-rebase
   in two states (`BLEND` + `IDLE_CLIP`). Update the architecture
   table to reflect the third state (`HOLD`) and the new
   `rebase_hold_msg` helper as part of the pose-pipeline shared
   library.
