# G1 ONNX Feasibility Sweep — Implementation, Usage & Results

Companion to [`g1_sonic_generated_x2_corpus.md`](g1_sonic_generated_x2_corpus.md) (the
experiment plan). This documents the **working implementation**: how the stock G1
SONIC ONNX is driven through the vectorized IsaacLab `im_eval` sweep to score the
bones-seed G1 corpus for feasibility + tracking quality, and the validation results.

Status: **mechanism validated end-to-end** (2026-07-11). Known-good G1 walks track
4/4; a 100-clip easy/hard sample discriminates correctly. Ready to scale.

---

## 1. What the released model is (and the key gotchas)

The G1 SONIC policy ships **only as an encoder+decoder ONNX pair**
(`gear_sonic_deploy/policy/release/model_{encoder,decoder}.onnx`) — there is **no
gear_sonic-native `.pt`**. So `eval_agent_trl` / `im_eval` (which normally
`torch.load` a `UniversalTokenActor`) cannot load it directly; we inject an
onnxruntime shim as `model.policy` instead.

ONNX I/O (byte-verified):

| model   | input `obs_dict` | output |
| ------- | ---------------- | ------ |
| encoder | `[N, 1762]`      | `encoded_tokens [N, 64]` |
| decoder | `[N, 994]`       | `action [N, 29]` |

Findings that make it work (these **correct** the plan's "framework-native, no
adapter" assumption):

- **The encoder wants the deploy's *joint-space* obs**, not the framework's default
  Cartesian `command_multi_future`. But for the **g1 encoder mode (id 0)** the deploy
  zero-fills every term except 4 required ones, so only these are populated:
  `encoder_mode_4` (= `[0,0,0,0]` for g1), `motion_joint_positions_10frame_step5`,
  `motion_joint_velocities_10frame_step5`, `motion_anchor_orientation_10frame_step5`.
  All three exist in the framework (`command_multi_future_joint_pos`,
  `joint_vel_multi_future`, `motion_anchor_ori_b_mf`), and `dt_future_ref_frames *
  target_fps = 0.1 * 50 = 5` matches the deploy's `10frame_step5`.
- **Decoder proprioception (930) = the framework `actor_obs` fed *verbatim*.** The
  `policy` obs group is emitted in exactly the deploy order
  `[base_ang_vel | joint_pos | joint_vel | actions | gravity_dir]` (×10-frame
  history) — do **not** reorder it. (Feeding it in the yaml `defaults` order instead
  makes the robot fall at step ~12.)
- **No joint remap inside IsaacLab.** The ONNX operates in IsaacLab/URDF joint order;
  the deploy only remaps at its hardware motor I/O.
- **Batch=1 export → dynamic batch.** The encoder bakes batch=1 into internal
  `Reshape` targets; the shim rewrites the leading dim to `-1` (verified bit-identical
  to batch=1, and batched == per-row on realistic obs).
- **bones-seed G1 CSVs are 120 fps** (not 50). Convert with
  `--fps 120 --fps_source 120` and let the eval loader resample 120→50. Using
  `--fps 50 --fps_source 120` silently skips downsampling (2.4 is a non-integer
  stride) → 2.4× slow-motion.

---

## 2. Tooling (all under `gear_sonic/scripts/`)

- **`g1_onnx_policy_shim.py`** — `G1OnnxPolicyShim`, the onnxruntime drop-in for
  `model.policy`. Builds the 1762 encoder vector (3 g1-mode terms from the command
  manager) and the 994 decoder vector (`[tokens | actor_obs]`), runs both ONNX,
  returns 29-DOF actions in IsaacLab order. Single-frame/stateless
  (`init/clear_rollout` are no-ops). Set `G1_SHIM_RECORD_DIR=<dir>` to dump each
  env's executed pose per step as a soma G1 CSV (same format as the input clips).
- **`run_g1_onnx_im_eval.py`** — launcher. Injects the shim into `eval_agent_trl`
  without a `.pt` (monkeypatch + throwaway checkpoint), reusing the `sonic_release`
  config from an existing run dir. Runs the `im_eval` callback headless.
- **`motion_deviation.py`** — executed-vs-reference deviation (Metric 2): MPJPE
  global + root-relative, root drift, joint MAE, stride. FK via the G1 MJCF; resamples
  to time-align.
- **`feasibility_report.py`** — combines Metric 1 (feasibility) + Metric 2 into a
  per-clip CSV with a 4-way label and an aggregate distribution.

Supporting edits:
- `gear_sonic/envs/manager_env/mdp/observations.py` — added
  `command_multi_future_joint_vel` (future reference joint velocities).
- `gear_sonic/scripts/play_motion_mujoco.py` — derives the DOF count from the MJCF so
  it plays any robot (29-DOF G1, not just 31-DOF X2).

---

## 3. The 4-way label

Per **feasible** clip (feasibility = the episode did **not** terminate):

| label | condition | corpus action |
| ----- | --------- | ------------- |
| **CLEAN** | pose tracked (upper & lower root-relative MPJPE < 50 mm) **and** base stayed put (drift < max(0.5 m, 2× reference root travel)) | prime data |
| **BASE-MOBILE** | pose tracked but base drifted (e.g. steps to balance a heavy manipulation) | keep + flag |
| **POOR-POSE** | upright but can't reach the pose (high local MPJPE, e.g. deep crouch / crawl) | keep, low-fidelity |
| **INFEASIBLE** | terminated (fell / diverged) | drop |

The gate is **content-adaptive**: the drift threshold scales with the reference's own
travel, so locomotion clips are effectively judged on drift/stride while in-place
manipulation tolerates drift up to the 0.5 m floor and is judged on pose. The
bones-seed category is carried through for grouping only. Thresholds (50 mm / 0.5 m)
are first-pass defaults — calibrate against visual judgement on borderline clips.

---

## 4. How to run

Environment: **`env_isaaclab`** conda (needs `onnx` + `onnxruntime`; both present).
Viewer/metric steps use a MuJoCo env (`.venv_sim`).

```bash
# 1. Build a G1 motion-lib PKL from soma G1 CSVs
#    bones-seed = 120 fps; teleop captures (g1_recorded) = 50 fps
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input <csv_dir> --output motions.pkl --robot g1 --fps 120 --fps_source 120

# 2. Sweep: run the stock ONNX over all clips, record executed trajectories
G1_SHIM_RECORD_DIR=executed_csv \
python gear_sonic/scripts/run_g1_onnx_im_eval.py \
    --motion-file motions.pkl --num-envs 128 --eval-output-dir eval_out
#   -> eval_out/metrics_eval.json  (terminated / progress per clip = Metric 1)
#   -> executed_csv/<clip>.csv     (executed soma G1 CSVs)

# 3. Build the executed PKL (dump is 50 fps = control rate)
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input executed_csv --output executed.pkl --robot g1 --fps 50 --fps_source 50

# 4. Score: per-clip feasibility + deviation + label
python gear_sonic/scripts/feasibility_report.py \
    --ref-pkl motions.pkl --exe-pkl executed.pkl \
    --metrics eval_out/metrics_eval.json \
    --mjcf gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml \
    --cat-map category_map.json --out feasibility_report.csv
```

Note: `im_eval` steps all envs to the **longest** clip in the batch, so shorter clips'
executed dumps are **padded with held-final-pose frames**; `feasibility_report.py`
trims each executed clip to its true length (`round(ref_frames * 50 / ref_fps)`)
before scoring.

Visually inspect a clip (executed vs reference), MuJoCo kinematic viewer:

```bash
.venv_sim/bin/python gear_sonic/scripts/play_motion_mujoco.py \
    --motion executed/<clip>.pkl \
    --mjcf gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml
```

---

## 5. Validation results (2026-07-11)

**Parity — known-good G1 walks (`g1_recorded/walk_keyboard_00{1..4}`):** 4/4 feasible
(success 1.0, progress 1.0), user-confirmed visually. Deviation: root-relative MPJPE
~19 mm, joint MAE ~3°, **stride ~0.91 (≈9% understep)** — the honest "what the body
can actually do" signal that keeps future training from chasing un-reachable strides.

**Discrimination — 100-clip sample (40 easy walking + 60 hard stunts/sports/dance/
advanced):**

| label | overall | easy (n=40) | hard (n=60) |
| ----- | ------- | ----------- | ----------- |
| CLEAN | 63% | 33 (83%) | 30 (50%) |
| BASE-MOBILE | 3% | 0 | 3 |
| POOR-POSE | 25% | 4 | 21 (35%) |
| INFEASIBLE | 9% | 3 | 6 |

Feasible (not terminated) = 91%. Hard clips have **5× the poor-pose rate** and half the
clean rate of easy clips. Labels are semantically correct: INFEASIBLE = jumps /
handstands / fast dance (fall mid-clip); POOR-POSE = crawls / deep crouches / stretches
(complete but poor pose); BASE-MOBILE = footwork dances (pose tracked, base wanders).

**Full corpus — feasibility over all 37,968 planner clips (2026-07-12).** Metric-1
sweep (2048 envs, length-sorted, no recording), ~55 min, one process:

| category | n | infeasible | feasible | mean progress |
| -------- | ----- | ---------- | -------- | ------------- |
| locowalk | 18,036 | 1.4% | 98.6% | 0.990 |
| locomanip | 9,712 | 5.2% | 94.8% | 0.963 |
| locopost | 8,752 | **34.7%** | 65.3% | 0.742 |
| locobal | 1,468 | 6.8% | 93.2% | 0.952 |
| **total** | **37,968** | **10.2%** | **89.8%** | — |

A 500-per-category sample predicted every rate to within ~2% (1 / 3 / 32 / 6%), so the
sampling was sound. Infeasible motions are all sensible: handstands (fall at ~2%),
heavy two-hand manipulations, ground descents (sits / deep crouches / cartwheels),
jumps, single-foot balance — concentrated in **locopost** (35% infeasible; only 8%
CLEAN in the sample).

**Headline:** **~28% of the corpus (~10% infeasible + ~18% poor-pose) is wasted or
degraded gradient if trained on the raw retarget** — the tracker burns it chasing
targets the robot can't hit; the executed motion encodes the achievable version
(shallower crouch, realistic footwork, honest ~9% understep). The gain is concentrated
in posture/ground motions (locopost) and negligible for walking (locowalk 98.6% clean).

## 6. Recorded executed corpus (Phase-2 data)

Metric 1 needs no recording, but the *executed* trajectories (for Phase-2 G1→X2
retarget/fine-tune and full-corpus Metric 2) require the recorder (`G1_SHIM_RECORD_DIR`).
Two recorder correctness rules (both hard-won):

- **Write once per clip, at its length** — not a periodic full-CSV rewrite (that was
  O(clip-length²) I/O and made long dance-card clips crawl).
- **Key by the motion-lib's global `_curr_motion_ids`, not `cmd.motion_ids`** — the
  latter is env-local (0..num_envs-1) and constant across env-loops, so it mislabels
  clips once the sweep spans more than one env-loop.

To keep it bulletproof, run the recorded pass in **single-env-loop chunks**
(`num_envs ≥ chunk_size`, e.g. 2048 clips/chunk): every feasible clip reaches its length
within its chunk and flushes exactly once; there are no cross-loop reassignments to get
wrong, and it avoids a physics CUDA-assert seen at multi-loop transitions. Filter the
recorded CSVs to the feasible set (from `metrics_eval.json`) to get the clean,
dynamically-consistent training corpus.
