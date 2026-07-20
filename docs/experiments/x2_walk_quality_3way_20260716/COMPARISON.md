# 3-way feasibility comparison — 61 G1-teleop clips (IsaacLab im_eval)

| category | n | G1-stock | base-3k | FT-1144 |
|---|---|---|---|---|
| slow_walk | 16 | 16/16 (100.0%) | 16/16 (100.0%) | 16/16 (100.0%) |
| slow_walk_turns | 23 | 23/23 (100.0%) | 23/23 (100.0%) | 23/23 (100.0%) |
| slow_walk_back | 7 | 7/7 (100.0%) | 7/7 (100.0%) | 7/7 (100.0%) |
| walk | 11 | 11/11 (100.0%) | 11/11 (100.0%) | 11/11 (100.0%) |
| run | 4 | 3/4 (75.0%) | 3/4 (75.0%) | 3/4 (75.0%) |
| **overall** | 61 | 60/61 (98.4%) | 60/61 (98.4%) | 60/61 (98.4%) |

## Per-clip failures (terminated clips, progress at termination)

### G1-stock
- run_004 — progress 0.661

### base-3k
- run_004 — progress 0.064

### FT-1144
- run_004 — progress 0.106

## Overall rates

| sweep | success_rate | progress_rate |
|---|---|---|
| G1-stock | 0.9836 | 0.9944 |
| base-3k | 0.9836 | 0.9847 |
| FT-1144 | 0.9836 | 0.9853 |

## Run metadata (2026-07-16, local RTX 5090, env_isaaclab, num_envs=32, sequential)

- A (G1-stock): `run_g1_onnx_im_eval.py --motion-file gear_sonic/data/motions/g1_teleop_corpus_50fps.pkl --num-envs 32` (stock release encoder/decoder ONNX, G1 29-dof)
- B (base-3k): `eval_x2_isaacsim_onnx.py --onnx ~/x2_cloud_checkpoints/dance_v3_3k_full/exported/dance_v3_3k_g1.onnx checkpoint=~/x2_cloud_checkpoints/dance_v3_3k_full/dance_v3_3k.pt` on x2_g1teleop_50fps.pkl, im_eval callbacks, fine_tune_dataset.enable=false
- C (FT-1144): same as B with `~/x2_cloud_checkpoints/g1teleop_overnight/sonic/snapshots/exported/ft_1144_g1.onnx` (exported this session from last_iter001144.pt, max|onnx-pt|=1e-6 rad)
- No repo code modified; im_eval_callback.py wrist-link patch already committed.
- run_004 is the only failure in all three sweeps (fastest run clip). G1-stock survives to 66% progress; both X2 models fall early (6-11%).
