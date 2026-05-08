# Real-Deploy Tuning Configs

YAML presets for the post-policy tuning knobs `deploy_x2.sh` exposes when
running on a real X2 Ultra. Pulled into the deploy via:

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/conservative.yaml \
    --model ...   --motion ...
```

Each YAML key maps to a single `x2_deploy_onnx_ref` CLI flag. The
translator at `gear_sonic_deploy/scripts/tuning_config_to_args.py` reads the
file and emits the corresponding flags; `deploy_x2.sh` prepends them to its
arg list so anything the operator passes explicitly on the command line
still overrides the preset.

## Parity rule (read this first)

`--tuning-config` is **rejected in `sim` mode**.

Sim profiles (`parity`, `handoff`, `gantry`, `gantry-dangle`) exist to
guarantee bit-for-bit equivalence between the C++ deploy binary and the
Python reference (`gear_sonic/scripts/eval_x2_mujoco.py`) running in
MuJoCo. Allowing a tuning config to silently change safety clamps or
filtering inside a sim run would erode that parity surface. The wrapper
exits with an error if you try.

If you want to *test* a real-deploy preset's effect in MuJoCo, do it via
explicit CLI flags (`--max-target-dev`, `--target-lpf-hz`, etc.) and accept
that the resulting trajectory **will not** match `eval_x2_mujoco.py`.

The post-policy filters that ship today (e.g. `target_lpf_hz`) are
*architecturally* parity-safe even on the real robot:

* The policy receives the same observation it always has -- nothing the
  YAML can set is wired into the obs builder.
* `--obs-dump` returns from the OnControl tick **before** the LPF runs,
  so `compare_deploy_vs_python_obs.py` is bit-identical with the filter
  on or off.
* RAMP_OUT and SAFE_HOLD bypass the LPF -- those states already produce
  a deliberately shaped trajectory we don't want to attenuate.

## Shipped presets

| Preset                | Use when                                                                                                    | Highlights                                                  |
|-----------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `conservative.yaml`   | First powered run with a new checkpoint OR new motion playlist                                              | `max_target_dev=0.30`, `target_lpf_hz=0`                    |
| `expressive.yaml`     | After conservative passes; want to actually see the reference's full arm range and tame leg/waist jitter    | `max_target_dev=0.80`, `target_lpf_hz=8`                    |

## Adding a new preset

1. Copy `conservative.yaml` to `<your_preset>.yaml`.
2. Update the `description:` block so the next operator can tell at a
   glance why this preset exists.
3. Tweak knobs. Run `python3 gear_sonic_deploy/scripts/tuning_config_to_args.py
   <your_preset>.yaml --validate` to confirm the file parses.
4. Use it: `deploy_x2.sh local --tuning-config <your_preset>.yaml ...`.

If you need a knob that isn't in the schema, see the "Adding a knob"
section in `_schema.yaml`. The translator rejects unknown keys explicitly
so a typo in a preset surfaces as a launch-time error rather than as
silently-ignored config.

## How presets compose with explicit CLI flags

```bash
deploy_x2.sh local \
    --tuning-config configs/real_deploy_tuning/expressive.yaml \
    --max-target-dev 0.50 \
    ...
```

The preset says `max_target_dev: 0.80`, but the explicit
`--max-target-dev 0.50` on the command line wins (it appears later in the
binary's arg list). This is the intended way to do quick A/B sweeps off a
known-good preset without copying the YAML.
