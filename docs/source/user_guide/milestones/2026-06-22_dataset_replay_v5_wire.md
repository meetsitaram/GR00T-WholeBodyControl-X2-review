# 2026-06-22 — Dataset replay: v5 future-window wire + single-shell launcher

> **Session focus.** Make `replay_x2_dataset.py` actually move the
> robot's body during playback (not just the OmniHand fingers), by
> promoting its wire payload to the deploy's v5 future-window
> contract. Ship a single-shell stack launcher
> ([`run_x2_replay_stack.sh`](../../../../gear_sonic/scripts/run_x2_replay_stack.sh))
> that mirrors the
> [`run_x2_vla_runtime.sh`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh)
> and
> [`run_x2_pkl_planner_stack.sh`](../../../../gear_sonic/scripts/run_x2_pkl_planner_stack.sh)
> pattern (deploy + foreground client + trap-cleaned shutdown), plus
> an opt-in `--with-rerun` flag that spawns the recorded-cameras
> viewer alongside the live deploy. Closes the v0 sketch + "roadmap"
> note in tutorial
> [§6.3](../../tutorials/x2_dataset_record_and_replay.md#63-replay-a-recorded-episode-through-the-live-deploy).

---

## TL;DR

| Symptom (before) | Cause | Fix |
|---|---|---|
| Replaying a recorded episode through the live deploy made the **OmniHand fingers move correctly** but the **body stayed in `idle_stand`** — feet planted, arms locked at default angles, only the fingers tracking the recording. The same parquet, kinematic-replayed through `replay_x2_kinematic.py`, showed the full body motion. So the data was fine; the wire wasn't. | The C++ deploy **ignores `motion_token` from the wire** and re-tokenizes the trajectory each tick from `joint_pos_mj_future` (the 9-slot lookahead window). The v0 replay published only the v4 envelope (current `joint_pos_mj` + `motion_token` + hand joints), with no future-window fields. The deploy's SUB therefore back-filled the future window with the trained `default_angles` stand pose, the encoder+FSQ+decoder ONNX re-tokenized THAT, and the body held idle. The OmniHand fingers worked because they have their own ZMQ pose-streamer that reads `left_hand_joints` / `right_hand_joints` from the wire directly — that path has no future-window dependency. | Promote the replay payload to v5 by adding the 5 future-window fields (`joint_pos_mj_future`, `root_quat_xyzw_future`, `joint_vel_mj_future`, `frame_index_future`, `future_dt_s`) built from the parquet's body_q at offsets `f+5, f+10, ..., f+45` (= 0.1 s spacing at the dataset's 50 Hz native fps), tail-tiled with the final frame for indices past episode end. Mirrors what [`live_vla_publish_motion_token`](../../../../gear_sonic/scripts/live_vla_publish_motion_token.py) ships every tick. |
| No single-shell launcher for replay. Operator had to start a sim deploy in one terminal, then run the replay python in another, then remember Ctrl-C order and `docker stop` the orphaned sim container themselves. The sibling Quest 3 / pkl / VLA stacks all had wrappers; replay was the odd one out. | Replay was historically a "sketch" workflow (recipe 6.3 in the tutorial), not a first-class verb. | New [`gear_sonic/scripts/run_x2_replay_stack.sh`](../../../../gear_sonic/scripts/run_x2_replay_stack.sh) mirrors the [`run_x2_pkl_planner_stack.sh`](../../../../gear_sonic/scripts/run_x2_pkl_planner_stack.sh) pattern: spawns `deploy_x2.sh sim --vla` (or skips it in `--no-deploy` / `--pc2-host` mode), waits for the deploy's `Launching ...` log marker, runs `replay_x2_dataset` as the foreground client, and tears everything down in reverse order on `Ctrl-C` / `EXIT` / `TERM`. The replay gets `SIGINT` first so its `hold_on_exit` ramp-down completes against a still-alive deploy before the sim container goes away. |
| `--sim-profile parity` (defaulted by the pkl wrapper) bailed at deploy spawn with `Error: --sim-profile parity requires --motion <pkl|yaml>`. | Parity mode RSIs the bridge to motion frame 0 of a baked PKL; replay has no such PKL and doesn't need one (the recorded `body_q` is its own ground truth). | Default the replay wrapper to `--sim-profile handoff` instead — the documented "final sim gate before powered runs" profile: bridge boots at `DEFAULT_DOF` (matches real-robot MC handoff), elastic band stays on through the 2 s soft-start ramp so the body cannot tip while the replay's 3 s `--countdown` warm-up transitions to frame 0 of the recording. |
| No way to eyeball the recorded cameras alongside the live deploy without running a separate `rerun` viewer command, manually picking the same `--episode`, and remembering the dedicated `.venv-viewer/` interpreter wrapper. | The two tools weren't wired together. | New `--with-rerun` flag on the stack wrapper spawns [`view_x2_recorded_dataset.sh`](../../../../gear_sonic/scripts/view_x2_recorded_dataset.sh) as Step 0/2 (BEFORE the deploy) so its 5–30 s cold-load runs in parallel. The rerun GUI process is intentionally spawned by `rr.init(spawn=True)` and **outlives the wrapper** so the operator can scrub the recording after the live run ends. |

---

## Architecture

### Wire promotion: v4 → v5

The C++ deploy on PC2 (`agi_x2_deploy_onnx_ref`) treats any pose frame
on `:5556` as v5-capable iff **both** `joint_pos_mj_future` and
`root_quat_xyzw_future` are present in the header's field list. With
v5 active, the deploy's fused encoder+FSQ+decoder ONNX re-tokenizes
the wire's future window each tick and tracks that trajectory. Without
them, it back-fills the future window with `default_angles` and the
body holds `idle_stand`.

```
                wire layout (per pose frame, every 20 ms)
                ────────────────────────────────────────
shared with                  v5 promotion fields (NEW)
v4 envelope                  ──────────────────────────
─────────────                joint_pos_mj_future   f32 (9, 31)
joint_pos_mj    f32 (31,)    root_quat_xyzw_future f32 (9, 4)
root_quat_xyzw  f32 (4,)     joint_vel_mj_future   f32 (9, 31)
motion_token    f32 (64,)    frame_index_future    i64 (9,)
left_hand_joints  f32 (10,)  future_dt_s           f32 (1,)
right_hand_joints f32 (10,)
frame_index     i64 (1,)              ▲
                                      │ presence of
                                      │ joint_pos_mj_future +
                                      │ root_quat_xyzw_future
                                      │ flips deploy → v5 mode
                                      ▼
                ───────────────────────────────────────
                  C++ deploy SUB :5556
                  └─► encoder+FSQ+decoder ONNX:
                        re-tokenize from FUTURE window
                  └─► motor commands @ 50 Hz
```

The 9 future slots span 0.5 s at 0.1 s spacing each — same cadence
the live VLA bridge ships, same cadence the trained model was
calibrated against. For the replay path the slots are sourced from
the recorded `body_q_mj` array at offsets `f+5, f+10, ..., f+45` (= 5
source frames per slot at 50 Hz native), tail-tiled with the final
frame when the lookahead goes past the end of the episode.
`joint_vel_mj_future` is the per-slot finite-difference of
`joint_pos_mj_future` divided by `future_dt_s`, with the first slot
diff'd against the current frame (`body_q[f]`) — same trick the
bridge's `_idle_future_payload_fields()` uses so the deploy can skip
its backward finite-diff path.

`root_quat_xyzw_future` is identity across all 9 slots because the
parquet records only `action.body_q_mj` (joint angles) and no per-frame
base orientation — same convention as the live VLA bridge, which
treats `root_quat` on the wire as the heading REFERENCE only and lets
the deploy fuse against its measured `base_quat` from `x2_debug`.

### Stack topology (sim mode)

```
LAPTOP                                       LAPTOP (docker)
══════                                       ═══════════════
                                             ┌──────────────────┐
replay_x2_dataset.py ─── pose :5556 ────────►│  deploy_x2.sh    │
(reads parquet,                              │  sim --vla       │
 builds v5 envelope                          │  + MuJoCo viewer │
 from body_q + f)                            │  + SONIC ONNX    │
        │                                    └──────────────────┘
        │                                             │
        │                                             ▼
        │                                       motor cmds @ 50 Hz
        │                                       (sim body tracks
        │                                        recorded trajectory)
        │
        └─ optional: --with-rerun
                ▼
        ┌────────────────────────────┐
        │  view_x2_recorded_dataset  │  reads same dataset/episode,
        │  (.venv-viewer)            │  spawns rerun GUI:
        │                            │   - 2x2 recorded MP4 grid
        │                            │   - 3-D wrist FK trace
        │                            │   - per-joint scalar timeline
        └────────────────────────────┘
              ▲
              │ GUI process outlives the wrapper -- you can scrub
              │ after the live run ends
```

### Stack topology (real-robot mode, `--pc2-host`)

```
LAPTOP                                       PC2 (real robot)
══════                                       ════════════════
                                             ┌──────────────────┐
replay_x2_dataset.py ─── pose :5556 ═wifi═══►│  x2_pc2_daemons  │
                                             │  - pose proxy    │
                                             │  - deploy --vla  │
                                             │  - hand bridge   │
                                             │  - motor monitor │
                                             └──────────────────┘
        + optional --with-rerun                       │
        (same rerun GUI as sim)                       ▼
                                              real motors @ 50 Hz
```

The replay's PUB binds locally on `${PUB_BIND_HOST}:5556` — the
host is gated on `--pc2-host` (see the
[2026-06-23 LAN-isolation follow-up](2026-06-23_pose_pub_lan_isolation.md)).
Without `--pc2-host`, the bind is `127.0.0.1` so PC2's always-on
pose proxy can't reach the wire over wifi even if its daemons are
still up from a previous real-robot session. With `--pc2-host`, the
bind is `*` so PC2 can SUB. PC2's pose proxy connects out from the
daemon side (assumed already running via
`./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start`).

---

## Operator workflow

### Sim smoke (one shell, no headset, no policy)

```bash
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0
```

Wrapper output:

1. Banner showing dataset / episode / rate / deploy / ports.
2. Step 1/2: spawn `deploy_x2.sh sim --vla --sim-with-omnihand
   --sim-viewer --wrist-bypass ik --sim-profile handoff
   --deploy-extra-arg --disable-pose-ref-watchdog`. Wait for the
   `Launching ...` log marker, settle 2 s.
3. Step 2/2: spawn `replay_x2_dataset`. Wait for the `PUB bound on`
   log marker.
4. Live: replay's 3 s countdown holds frame 0 (giving the deploy's
   handoff ramp time to settle on the recording's starting pose),
   then the trajectory plays at 50 Hz, then 0.5 s of `hold_on_exit`
   on the last frame.
5. Replay exits rc=0; reverse-order shutdown: `SIGINT replay` (no-op,
   it's already done), `SIGINT deploy host-bash`, `docker stop` the
   sim container, exit rc=0.

### Sim + recorded cameras side-by-side

```bash
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0 --with-rerun
```

Same as above, plus Step 0/2 spawns the rerun viewer (in the dedicated
`.venv-viewer/`). The rerun GUI loads the 4 recorded camera MP4s
(`ego_view`, `head_front`, `stereo_left`, `stereo_right`), the
wrist-FK 3-D trace, and the per-joint scalar timeline for the same
episode. The MuJoCo viewer shows the live deploy executing the
matching `body_q_mj`. The rerun GUI stays open after the wrapper
exits so the operator can scrub the recording for comparison.

### Real-robot first pass (half-speed, e-stop in reach)

```bash
# On PC2 (separate shell, separate machine):
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start \
    --attach --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight

# On the laptop:
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0 \
    --pc2-host 192.168.86.32 --rate-scale 0.5 --with-rerun
```

`--rate-scale 0.5` halves the effective publish rate (50 Hz → 25 Hz);
the future-window stride stays at 5 source frames per slot (i.e. each
slot still represents 0.1 s of native recording dynamics, not 0.2 s
of slowed wall-clock). The 9 slots therefore still carry the same
physical lookahead the deploy's tokenizer was trained on, regardless
of `--rate-scale`.

### Bypass / inspect

```bash
# Dry-run: load the parquet, print stats, do NOT bind ZMQ.
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0 \
    --no-deploy --dry-run

# Free ports + kill orphan deploy/replay processes from a crashed run.
./gear_sonic/scripts/run_x2_replay_stack.sh --cleanup-only

# Run replay alone against an externally-managed deploy.
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0 --no-deploy
```

---

## Tests added

| Test | What it pins |
|---|---|
| `tests/test_replay_x2_dataset_future_window.py::test_future_window_basic_indexing` | At `f=0, step=5` the 9 slots are `body_q[5, 10, ..., 45]`. Synthetic ramp `body_q[i, :] = i`. |
| `…::test_future_window_basic_indexing_mid_episode` | At `f=20, step=5` the slots are `body_q[25, 30, ..., 65]`. Catches off-by-one in the offset base. |
| `…::test_future_window_tail_tiles_past_episode_end` | Indices past `n_frames` clamp to the last frame. Two cases: f=10/step=5 (all slots tail-tiled) and f=5/step=2 (partial tail). |
| `…::test_future_window_velocity_finite_diff_against_current_frame` | `joint_vel_mj_future[0] = (body_q[f+step] - body_q[f]) / dt`, not `(body_q[f+step] - body_q[f+step])` (a bug we'd otherwise ship). |
| `…::test_future_window_velocity_zero_when_tail_tiled` | Tail-tiled slots have inter-slot vel = 0 (no change between identical frames). Slot 0 still has the diff against the current frame. |
| `…::test_future_window_step_zero_raises` | `step=0` raises a clear `ValueError` instead of a silent broadcast bug. |
| `…::test_build_payload_schema_matches_v5_contract` | Exact-key + exact-dtype + exact-shape match against the deploy's v5 promotion contract (every one of the 11 fields). |
| `…::test_build_payload_current_frame_sourced_from_f` | Current-frame `joint_pos_mj` slices into `body_q[f]`, not `body_q[0]` or `body_q[-1]`. |
| `…::test_build_payload_clamps_f_past_episode_end` | Replay's hold-on-exit passes `f = last_f`; if the caller accidentally passes `f >> n_frames-1`, the payload still produces a valid envelope (current + tail-tiled future). |
| `…::test_build_payload_wire_frame_indexing` | `frame_index = [wire_frame]` and `frame_index_future = wire_frame + [1..9]` (monotonic across the replay run, even across the warm-up → trajectory → hold-on-exit boundaries). |
| `…::test_build_payload_root_quat_identity_for_current_and_future` | Replay has no recorded base orientation; both current and future `root_quat_xyzw` are identity (consistent with the bridge's convention). |
| `…::test_build_payload_future_dt_matches_module_constant` | `future_dt_s = [0.1]` matches `FUTURE_DT_S` exactly (catches accidental drift between the helper and the wire field). |
| `…::test_payload_packs_and_unpacks_byte_for_byte` | **The single most important test in the file.** `pack_pose_message(..., version=4)` → `unpack_message()` roundtrips every byte of every v5 field. If the deploy's decoder can read what we pack, the deploy will promote the frame to v5 and the body will track. Uses non-trivial random values so any swap/transpose bug surfaces. |

13/13 tests green; all 38 sibling replay tests (`test_replay_x2_kinematic.py`, `test_replay_finger_curl_comparison.py`) continue to pass.

---

## What this milestone does **not** do

- It does not change the C++ deploy in any way (the v5 contract was always present; we just weren't filling it on the replay path).
- It does not change `replay_x2_kinematic.py` or the offline retargeting replay flow — both were always wire-free.
- It does not change any wire-shaping numbers on the live VLA bridge (LPF cutoffs, ramp ticks, step caps, etc.).
- It does not change the `motion_token` on the wire. The replay still copies `action.motion_token` from the parquet into the envelope; the deploy still ignores it. The token is a debug echo, not the source of truth.
- It does not add live PC2 cameras to the wrapper. The stack has [`gear_sonic/scripts/run_camera_viewer.py`](../../../../gear_sonic/scripts/run_camera_viewer.py) for that already; bolting it onto the replay wrapper is a small follow-up.

---

## Files

**Created:**

- `gear_sonic/scripts/run_x2_replay_stack.sh` — single-shell launcher (deploy + replay + optional rerun), reverse-order trap cleanup, three modes (sim default / `--no-deploy` / `--pc2-host`).
- `tests/test_replay_x2_dataset_future_window.py` — 13 tests covering the future-window helper, the v5 payload schema, and the pack/unpack byte-roundtrip.
- `docs/source/user_guide/milestones/2026-06-22_dataset_replay_v5_wire.md` — this file.

**Modified:**

- `gear_sonic/scripts/replay_x2_dataset.py` — new `_build_future_window()` helper + `FUTURE_*` constants; `_build_payload()` refactored to take the whole-episode arrays + a source-frame index + the wire-frame counter + the future-step, and to populate the 5 v5 promotion fields; `main()` derives `future_step` from the dataset's native fps (independent of `--rate` / `--rate-scale`); updated module docstring documenting the v5 contract + why it's required.
- `docs/source/tutorials/x2_dataset_record_and_replay.md` — replaced the §6.3 sketch + "roadmap" admonition with the real `run_x2_replay_stack.sh` workflow + a callout to this milestone; §6.5 matrix's "SONIC replay" cell now points at the new wrapper instead of "planned"; added the new test file to the §7 acceptance gates list.
- `gear_sonic/scripts/run_x2_vla_runtime.sh` + `gear_sonic/scripts/run_x2_pkl_planner_stack.sh` headers — added `run_x2_replay_stack.sh` to the launcher-family cross-reference table so all three siblings link to each other.
- `sample_commands.md` — new "SONIC-loop replay (full deploy + future-window wire)" subsection under "Replay".

---

## Follow-ups

- Bolt the live PC2 camera viewer (`run_camera_viewer.py`) onto the wrapper as `--with-live-cameras` so real-robot runs can show "what the recording looked like" (rerun GUI) AND "what the robot is seeing right now" (OpenCV) side-by-side.
- Synchronize the rerun viewer's playback timeline to the live replay's `wire_frame` so the recorded MP4s scrub forward in lockstep with the live deploy (today they're a static reference; the operator scrubs manually). Likely a Python sidecar that SUBs the replay's `pose` wire and PUBs `rr.set_time_sequence` into the same recording_id.
- Add a `--episode-range N M` knob so the wrapper can sequence through multiple episodes back-to-back without operator intervention (handy for batch acceptance sweeps).
- Bake the v5 future-window helper into a shared utility module so the live VLA bridge and the replay tool stop maintaining two near-identical implementations. The bridge's `_idle_future_payload_fields()` is the obvious target.
