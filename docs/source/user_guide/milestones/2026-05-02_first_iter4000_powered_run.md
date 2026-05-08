# 2026-05-02 — First iter-4000 X2 Ultra Powered Run

> **A cherished moment.** First powered run on the real X2 Ultra after
> closing the C++ sim-to-sim parity gap. Robot stood **rock solid** for the
> full 5-second control window, completed RAMP_OUT cleanly, and handed
> back to MC with no red-light fault. Zero action-clip events.

---

## The run

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model  ~/x2_cloud_checkpoints/h200-iter-4000-20260501/model_step_004000.onnx \
    --motion ./gear_sonic/data/motions/playlists/minimal_v1.yaml \
    --autostart-after 5 --max-duration 5 \
    --max-target-dev 0.30 --ramp-seconds 2.0 --tilt-cos -0.3 \
    --return-seconds 2.0 \
    --log-dir /tmp/minv1_$(date +%Y%m%d_%H%M%S)
```

A single command, from the host shell, no manual docker dance, no manual
ROS sourcing. The script noticed it wasn't inside the `docker_x2/x2sim`
container, mounted `$HOME` at `$HOME`, cleared the sim-mode DDS isolation,
re-execed itself inside the container, ran the full pre-flight, asked
twice for operator confirmation, and launched.

## What the deploy saw, second by second

```
[INIT -> WAIT_FOR_CONTROL]   safe-hold latched at current observed pose
[5.00 s autostart]           Reference motion yaw-anchored to robot heading
                             (robot yaw = 8.99 deg, applied Δyaw = 9.00 deg)
[CONTROL tick=050]  policy_t=1.00s  alpha=0.49  grav_z=-0.98
[CONTROL tick=100]  policy_t=2.00s  alpha=0.99  grav_z=-1.00
[CONTROL tick=150]  policy_t=3.00s  alpha=1.00  grav_z=-1.00
[CONTROL tick=200]  policy_t=4.00s  alpha=1.00  grav_z=-0.99
[CONTROL tick=250]  policy_t=5.00s  alpha=1.00  grav_z=-0.99
[Max duration elapsed]  -> RAMP_OUT (2.00s return-to-default)
[RAMP_OUT complete]     -> shutting down
[cleanup]               MC start_app POSTed
```

**Headline numbers:**

| | Value |
|---|---|
| Time upright | 5.00 s of CONTROL + 2.00 s of RAMP_OUT, all clean |
| `grav_z` over the run | -0.98 to -1.00 (perfectly upright) |
| Action-clip events | **0** |
| `max_pre_clip` | 0.00 (policy nowhere near the 20.0 rad safety clamp) |
| Tilt watchdog | did not trip |
| MC handoff | clean (no red-light fault) |

## The journey here

This was not a fast win. The path from "ONNX sort-of works" to "robot
stands cleanly on hardware" took three layers of bug fixing, all of
which had to land before this command would do what it does today:

1. **G20 (2026-05-01):** Deploy-side 6D rotation channel-order bug.
   `motion_anchor_ori_b_mf_nonflat` was column-major
   `concatenate([col0, col1])` instead of the IsaacLab-canonical
   row-major `mat[:, :2].reshape(-1)`. Symptom on hardware: robot
   turns ~180° in the first 1-2 s and walks the wrong way. After the
   fix, sim yaw drift stays under 10°.

2. **G21 (2026-05-01):** ONNX-side tokenizer layout bug. The
   `eval_x2_mujoco_onnx.py::OnnxActor` wrapper was applying a
   spurious "interleaved → grouped" rearrangement on the 680-D
   tokenizer slice before the ONNX session. Pre-fix `--compare-pt`
   reported 3+ rad action delta over a rollout while PyTorch alone
   was faithful. Fix: remove the rearrangement. The fused g1 ONNX
   expects per-frame interleaved layout — same as
   `build_tokenizer_obs` already produced.

3. **G22 (2026-05-02):** Sim bridge handshake leak. The C++
   `OnWriter` (500 Hz) was publishing a default-zero `latest_cmd_`
   the moment FSM hit `CONTROL`, before `OnControl` had ever
   populated it. The bridge took that as the "first command", flipped
   its pre-handoff freeze off, and let `mj_step` integrate ~15 ms of
   zero-PD free-fall before the policy ever observed the RSI state.
   Fix: gate `OnWriter` behind a `latest_cmd_ready_` atomic that
   flips only after a real command has been latched.

4. **G23 (2026-05-02):** Sim bridge IMU read-side qvel parity. The
   bridge's IMU publisher was reading angular velocity via
   `mj_objectVelocity(BODY, pelvis_id, ang, flg_local=1)`, which
   returns the angular component of `mj_data.cvel` (CoM-frame
   spatial velocity). After a manual `qvel` write + `mj_forward`
   that does NOT round-trip exactly to `qvel[3:6]`. The deploy's
   `base_ang_vel` was OOD by ~0.02 rad/s on every IMU publish.
   Fix: when IMU body is the floating-base body, publish
   `qvel[3:6]` directly.

After G22 + G23, the C++ deploy's first-tick observation in
`--sim-profile parity` is **bit-exact** against
`eval_x2_mujoco_onnx.py --obs-dump`:

| Field | Pre-fix max\|Δ\| | Post-fix max\|Δ\| |
|---|---|---|
| `joint_pos_mj`     | 6e-4   | **0.000000** |
| `joint_vel_mj`     | 0.0169 | **0.000000** |
| `tokenizer_obs`    | 8.7e-5 | **0.000000** |
| `base_ang_vel`     | 0.0196 | **0.000000** |
| `proprioception`   | 0.0196 | **0.000000** |

In simulation, all three sim profiles (`parity`, `handoff`, `gantry`)
clear the 30 s standing gate with `grav_z = -1.00` flat throughout
and clean RAMP_OUT.

On hardware, the iter-4000 H200 checkpoint did exactly what the
training run promised: it stood.

## Operator-experience improvements that landed alongside

These don't change the policy, but they made the bring-up tractable
and they're worth remembering:

- **`x2_preflight.py` is now wired into Step 1/4** of `deploy_x2.sh`.
  Twelve checks (4 joint-group state topics, IMU, MC presence,
  arm-cmd conflict, IMU stillness + upright, joint pose / vel /
  effort) run automatically before the MC stop. A failing preflight
  aborts while MC is still holding the robot.

- **Auto-relaunch into `docker_x2/x2sim`.** No more manual
  `cd docker_x2 && docker compose run --rm --service-ports x2sim
  bash -lc "source ROS && ..."` incantation. The script detects a
  host shell and re-execs with `$HOME` mounted at `$HOME` so paths
  like `~/x2_cloud_checkpoints/...` and `./gear_sonic/...` resolve
  identically inside.

- **`--sim-profile {parity,handoff,gantry}`** as named, validated
  test scenarios. `parity` is the bit-for-bit Python eval mirror
  used as the diagnostic harness for G22/G23. `handoff` mimics the
  real-robot MC handoff with a soft-start ramp from
  `gantry_hang` pose. `gantry` is band-supported standby. All three
  are now part of the CI gate.

- **Two safety gates.** Gate 1/2 prompts before the MC stop and
  installs a cleanup trap that re-`POST start_app` on exit
  (Ctrl-C-safe). Gate 2/2 prompts before the actual deploy launch
  with the full resolved command line printed for review.

## What this run does NOT prove yet

Read this before pushing the envelope:

- **Active-balance behaviour is still untested on hardware.** The
  policy held the standing pose, but no perturbation was applied.
  Push-recovery and dynamic motion validation is the next ladder.
- **`--max-target-dev 0.30 rad` was not even close to saturating**
  (`act_clip_ticks=0`, `max_pre_clip=0.00`). Lift this gradually
  toward IsaacLab's measured peak (~1.30 rad on `take_a_sip`).
- **5 s is just the operator-confidence floor.** Walk
  `--max-duration` up the ladder: 5 → 10 → 30. The sim gate already
  cleared 30 s on all three profiles; hardware should follow.
- **Only `minimal_v1.yaml` was tested.** Move to
  `standing_gestures_v1.yaml` and `warehouse_v1.yaml` once the
  duration ladder is clean.

## Suggested ladder for the next session

```
Run 1-2:   repeat the exact command above (reproducibility)
Run 3:     --max-duration 10
Run 4:     --max-duration 30
Run 5:     --max-target-dev 0.50 (still 30 s)
Run 6:     --max-target-dev 0.80
Run 7:     --max-target-dev 1.00
Run 8:     --motion standing_gestures_v1.yaml
Run 9:     --motion warehouse_v1.yaml
Run 10:    light push-recovery test
```

## See also

- {doc}`../sim2sim_mujoco` — G22 and G23 write-ups, with the full
  pre/post numerical-validation tables and the lessons learned.
- {doc}`../x2_sim_to_real_wip` — running deploy status doc; the
  TL;DR was updated on 2026-05-02 to reflect the closed parity gap.
- `gear_sonic_deploy/scripts/compare_deploy_vs_python_obs.py` — the
  slot-by-slot obs differ that surfaced G22 / G23 in 30 minutes
  once we had the dump format right on both sides.

---

*"the python implementation is flawless and able to hold the robot
stably for the whole 30 seconds. if there is any issue, it has to be
in the c++ code"* — operator instinct, 2026-05-02.

It was. It is now fixed.
