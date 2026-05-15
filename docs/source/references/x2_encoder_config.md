# X2 Encoder Observation Configuration

This page documents the YAML schema that drives the X2 dataset
recorder's *inline* SONIC tokenizer -- the path that fills the
`action.motion_token` column in every recorded LeRobot frame so the
VLA has a real supervision target instead of a 64-D zero placeholder.

The config lives at:

```
gear_sonic/data/encoder/x2_observation_config.yaml
```

It is loaded by
[`X2EncoderObsBuilder`](#api-reference) (Python, recorder-side) and
its filename + selected mode are surfaced in the wrapper banner so an
operator can confirm at a glance which observation contract every
recorded episode pinned against.

---

## Why this file exists

The X2 deploy ONNX (`actor.onnx`) is a *PPO-fused* graph: the SONIC
encoder, the FSQ quantizer, and the policy decoder are all stitched
together inside one `.onnx` file. There is no separate
`encoder.onnx` the way G1 ships it, but the encoder weights still
exist independently inside the `.pt` checkpoint, under
`actor_module.encoders.g1.module.*`.

The recorder reuses those weights to label `action.motion_token`
*per-frame*. To produce the *same* token the deploy actor's internal
encoder would emit on the same wire snapshot, the recorder must feed
the encoder the *same* 680-D 10-frame future observation the
training pipeline used. That observation layout is what this YAML
pins.

Pinning the layout to YAML (instead of hard-coding it in Python)
gives us:

1. **Parity with the G1 release**: same vocabulary
   (`encoder_observations` + `encoder_modes`) so artifact diagrams
   read the same across both robots.
2. **An audit trail** when we change observation layout (the YAML
   diff shows up alongside the recorder change).
3. **A single chokepoint** the validation tooling
   (`compare_recorder_vs_deploy_obs.py`) can target to assert byte-
   parity between the recorder Python and the deploy C++ paths.

---

## Architecture: where the YAML fits

```mermaid
flowchart TB
  subgraph Operator
    Q[Quest 3 + WebXR]
  end
  Q -->|VR controllers| MGR[quest3_manager]
  MGR -->|locomotion_cmd| PLN[x2_heuristic_planner]
  PLN -->|body_pose +<br/>joint_pos_mj_future<br/>(31 + 9x31)| REC[x2_dataset_recorder<br/>subscribe mode]
  PLN -.->|joint_pos_mj +<br/>joint_pos_mj_future| DPL[deploy:<br/>x2_deploy_onnx_ref<br/>C++]
  MGR -->|arm_targets +<br/>hand_finger_cmd| REC
  REC -->|ZMQ pose topic<br/>(merged 31-D)| DPL

  subgraph Recorder["Recorder (Python)"]
    direction TB
    REC --> SNAP[snapshot dict]
    SNAP --> BLD[X2EncoderObsBuilder<br/>YAML-driven]
    BLD --> OBS[680-D encoder_input]
    OBS --> ENC[SONIC encoder<br/>extracted from .pt]
    ENC --> TOK[FSQ-quantized<br/>64-D motion_token]
    TOK --> PARQ[(action.motion_token<br/>in LeRobot parquet)]
  end

  subgraph Deploy["Deploy (C++)"]
    direction TB
    DPL --> ZPS[ZmqPoseInputSource]
    ZPS --> DEPOBS[680-D encoder_input]
    DEPOBS --> FUSED[Fused actor.onnx<br/>encoder + FSQ + decoder]
    FUSED --> JT[joint targets]
  end

  CFG[(x2_observation_config.yaml)] -.-> BLD
  CFG -.->|same contract,<br/>different consumer| FUSED
```

The YAML is the "single source of truth" arrow connecting the
recorder's encoder and the deploy's fused actor: both consume the
same 680-D observation; the recorder writes the encoder's output to
the parquet, and the deploy runs the same encoder weights internally
on the same observation.

---

## Schema

```yaml
encoder:
  dimension: 64               # FSQ-quantized motion-token output (2 x 32)
  motion_fps: 50.0            # Recorder + planner control rate (Hz)
  dt_future_ref: 0.1          # Future window stride (seconds)
  num_future_frames: 10       # 10 frames x 68 features = 680-D obs

  encoder_observations:
    - name: "x2_command_multi_future_nonflat"
      enabled: true

  encoder_modes:
    - name: "retargeted_body_q"
      mode_id: 0
      required_observations:
        - x2_command_multi_future_nonflat
```

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `encoder.dimension` | int | FSQ output dimension. The X2 encoder emits 2 tokens × 32 dims = 64. |
| `encoder.motion_fps` | float | Frame rate the recorder + planner run at. |
| `encoder.dt_future_ref` | float | Future-frame spacing in seconds. `10 frames × 0.1 s = 0.9 s` lookahead matches IsaacLab's `arange(NUM_FUTURE_FRAMES) * frame_skips`. |
| `encoder.num_future_frames` | int | Per-frame size: `31 jpos + 31 jvel + 6 ori = 68` features. Total = `10 × 68 = 680`. |
| `encoder.encoder_observations` | list | Superset of all observations any mode might consume. Each entry is a `{name, enabled}` mapping. |
| `encoder.encoder_modes` | list | Per-mode observation requirements. Each mode lists the subset of `encoder_observations` it needs. |

### Today's single observation

Only one observation is registered:

| Name | Dim | Description |
|---|---|---|
| `x2_command_multi_future_nonflat` | 680 | 10-frame future window of retargeted body_q + 6D rotation diff (mirrors `gear_sonic.scripts.eval_x2_mujoco.build_tokenizer_obs`). |

Adding a future modality (e.g. SMPL human pose) is a YAML + registry
edit -- no recorder rewrite. See the API reference below for the
registry contract.

### Today's single mode

| Name | Mode ID | Required observations |
|---|---|---|
| `retargeted_body_q` | 0 | `x2_command_multi_future_nonflat` |

When the recorder builds an obs, it dispatches through the registry
once per enabled observation in YAML order; concatenates the
results; and feeds the resulting 680-D vector to the SONIC encoder.

---

## How it differs from G1

G1 ships **two separate ONNX files** (`encoder.onnx` + `policy.onnx`)
and the encoder is **multi-modal at runtime**: its
`gear_sonic_deploy/policy/release/observation_config.yaml` declares
fourteen observations and three modes (`g1`, `teleop`, `smpl`). The
deploy switches modes at runtime to handle SMPL human-pose targets,
G1 retargeted motions, and VR-only sparse points.

X2 ships **one fused ONNX** with the encoder internal to the policy
graph. That fused graph was trained on *only* the `retargeted_body_q`
modality, so the YAML here pins exactly that one modality. The
schema is structured to accept future modalities without a recorder
rewrite -- if a future X2 release re-trains the encoder with SMPL
input, we add a `smpl_human_pose` entry to the registry + YAML and
the recorder handles it transparently.

---

## API reference

### `X2_OBSERVATION_REGISTRY`

```
X2_OBSERVATION_REGISTRY: Dict[str, GatherFn]
```

Maps observation name to a gather function with signature::

    def gather(snap, *, motion_fps, num_future_frames) -> np.ndarray

The default registry (in
`gear_sonic/utils/teleop/x2_encoder_obs_builder.py`) ships exactly
one entry. To add a new modality, append a function and a YAML entry
referencing it.

### `X2EncoderObsBuilder`

```python
from gear_sonic.utils.teleop.x2_encoder_obs_builder import X2EncoderObsBuilder

builder = X2EncoderObsBuilder.from_yaml(
    Path("gear_sonic/data/encoder/x2_observation_config.yaml")
)
obs_680 = builder.build_obs(snap, mode="retargeted_body_q")
```

The builder validates at construction time that:

* every `encoder_observations.name` is in the registry (typos fail
  loudly), and
* every `encoder_modes.required_observations` is a subset of the
  enabled observations (otherwise the runtime call would see an
  unset slot).

### `OnlineSonicTokenizer.from_checkpoint_with_config`

Recommended factory for the recorder. Loads the SONIC encoder
weights from a `.pt` and the gather config from this YAML. The
recorder calls it automatically when `--sonic-checkpoint` and
`--encoder-config` are both provided (the wrapper sets both by
default).

```python
from gear_sonic.utils.teleop.online_sonic_tokenizer import OnlineSonicTokenizer

tok = OnlineSonicTokenizer.from_checkpoint_with_config(
    "/path/to/model_step_025000.pt",
    "gear_sonic/data/encoder/x2_observation_config.yaml",
    device="cpu",
)
token_64 = tok.encode_with_snapshot(snap)
```

---

## Validation

The recorder ships a four-layer validation pyramid for the inline
tokenizer plumbing:

| Layer | Tool | What it checks |
|---|---|---|
| 1 | `pytest tests/test_x2_encoder_obs_builder.py` | Registry, YAML loader, builder shape/dtype contract |
| 1 + 2 | `pytest tests/test_x2_dataset_recorder_real_future_token.py` | Recorder chokepoints + parity vs `build_tokenizer_obs` and `label_trajectory` |
| 3 | `gear_sonic_deploy/scripts/compare_recorder_vs_deploy_obs.py` | Byte-parity between recorder Python obs and deploy C++ obs |
| 4 | `gear_sonic/scripts/validate_encode_decode_loop.py` | Encode round-trip + (optional) decoder consistency on a recorded parquet |

Layers 1 and 2 are CI-friendly (run on every push); layers 3 and 4
require a live planner stack and a recorded parquet respectively, so
they're documented incantations rather than automated gates.
