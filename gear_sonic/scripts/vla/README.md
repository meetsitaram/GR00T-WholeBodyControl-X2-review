# VLA (GR00T) fine-tune + autonomous inference

Everything in this folder is only needed when training or running the
GR00T VLA model on X2 datasets. **Live VR teleop and dataset recording do
not use these scripts** — the teleop/recording stack
(`run_x2_quest3_planner_stack.sh`, `record_x2_dataset.py`) is fully
self-contained without them.

| Script | Purpose |
| --- | --- |
| `train_groot_vla.sh` | One-command VLA fine-tune wrapper (pre-flight checks, background trainer, log tailing). |
| `launch_finetune_x2.py` | The underlying fine-tune entrypoint `train_groot_vla.sh` wraps (X2 embodiment + modality config wiring). |
| `run_x2_vla_runtime.sh` | Autonomous VLA inference runtime — sim or real robot (`--pc2-host`), optional teleop takeover (`--enable-takeover`). |
| `run_live_vla_demo.sh` | Deprecated forwarder to `run_x2_vla_runtime.sh`. |
| `inspect_vla_chunks.py` | Post-mortem tool: summarizes the `motion_token` chunk dumps the runtime writes. |

Two files here are **shared beyond VLA** (the teleop / recording / deploy
stacks use them too — they live here for naming consistency, but moving or
deleting them breaks non-VLA flows):

| Script | Shared role |
| --- | --- |
| `live_vla_publish_motion_token.py` | The core token→pose bridge. The Quest3 teleop stack spawns it (`-m gear_sonic.scripts.vla.live_vla_publish_motion_token`) and PC2 deploy scripts import from it. |
| `mock_vla_publish_stand_token.py` | Wire-format test publisher (referenced by deploy + planner constants). |

One more "vla"-named file stays outside on purpose:
`gear_sonic/data/features_x2_vla.py` — the LeRobot dataset feature schema,
imported by the dataset recorder (it belongs with the data definitions).
