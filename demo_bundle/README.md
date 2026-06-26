# Xbox demo bundle (demo_bundle)

Built at: `2026-06-26 02:35:27 PDT`
Total payload: **598.4 KB** across **20** files.

Self-contained bundle of every PKL referenced by `gear_sonic/scripts/play_xbox_controller.py` (locomotion + gestures + e-stop ack). The directory tree mirrors the repo layout, so you can rsync this folder directly on top of the `gear_sonic/` checkout on the demo machine without any path rewriting.

## Sync to the demo machine

Copy the whole bundle alongside the repo (preserves the
manifest + README) so you can re-verify on-site:

```bash
# on the build machine (this one):
rsync -avz --progress demo_bundle/ <user>@<demo-host>:/path/to/GR00T-WholeBodyControl/demo_bundle/

# on the demo machine, overlay the data into gear_sonic/
# (rsync without --delete so unrelated PKLs already on the
# demo box aren't wiped):
rsync -av demo_bundle/gear_sonic/ gear_sonic/
```

If the demo machine doesn't have the `gear_sonic` checkout at all, just clone the repo first and then run the second `rsync` above.

## Verify the payload on the demo machine

Re-hashes every file in the bundle against `manifest.json`. Run it from inside the bundle:

```bash
cd demo_bundle && python3 verify.py
```

Pass `--against <dir>` to verify the deployed gear_sonic tree instead (after the second rsync above):

```bash
python3 demo_bundle/verify.py --against /path/to/GR00T-WholeBodyControl
```

## Locomotion (D-pad + L2+R2 deadman, L1+R1 released)

| chord | PKL | size | duration | sha-256 |
|-------|-----|-----:|---------:|---------|
| `D-pad UP` | `gear_sonic/data/motions/x2_ultra_relaxed_walk_forward_v1.pkl` | 59.5 KB | 13.3 s | `7bc27ea0c5d8…` |
| `D-pad LEFT` | `gear_sonic/data/motions/x2_ultra_relaxed_walk_one_left_turn_v1.pkl` | 53.4 KB | 11.9 s | `8aff545ffb60…` |
| `D-pad RIGHT` | `gear_sonic/data/motions/x2_ultra_relaxed_walk_one_right_turn_v1.pkl` | 54.6 KB | 12.2 s | `961430ca4790…` |
| `D-pad DOWN` | `gear_sonic/data/motions/x2_ultra_relaxed_walk_two_right_turns_v1.pkl` | 88.7 KB | 19.8 s | `05107fb8b510…` |

## Gestures (A/B/X/Y bare or + single modifier L1|R1|L2|R2)

Any other shoulder/trigger combo silences face buttons (see launcher docstring + cheatsheet).

| chord | PKL | size | duration | sha-256 |
|-------|-----|-----:|---------:|---------|
| `A` | `gear_sonic/data/motions/x2_recorded/demo_gestures/hug3_001.pkl` | 47.6 KB | 16.6 s | `425cf2c3a8ac…` |
| `B` | `gear_sonic/data/motions/x2_recorded/demo_gestures/hand_on_shoulder_001.pkl` | 51.9 KB | 17.4 s | `094f1a63663f…` |
| `X` | `gear_sonic/data/motions/x2_recorded/demo_gestures/what_can_i_do_001.pkl` | 28.4 KB | 7.6 s | `1e9317ad229f…` |
| `Y` | `gear_sonic/data/motions/x2_recorded/demo_gestures/come_here_001.pkl` | 24.7 KB | 6.9 s | `30feddc827a4…` |
| `A+L1` | *(free)* | — | — |
| `B+L1` | *(free)* | — | — |
| `X+L1` | *(free)* | — | — |
| `Y+L1` | `gear_sonic/data/motions/x2_recorded/mc_gestures/bow_001.pkl` | 24.8 KB | 6.7 s | `5eebd9df3a76…` |
| `A+R1` | *(free)* | — | — |
| `B+R1` | *(free)* | — | — |
| `X+R1` | *(free)* | — | — |
| `Y+R1` | `gear_sonic/data/motions/x2_recorded/mc_gestures/right_shake_001.pkl` | 16.9 KB | 7.5 s | `b87668075586…` |
| `A+L2` | *(free)* | — | — |
| `B+L2` | *(free)* | — | — |
| `X+L2` | *(free)* | — | — |
| `Y+L2` | `gear_sonic/data/motions/x2_recorded/demo_gestures/left_wave_high_001.pkl` | 33.2 KB | 8.9 s | `6e7dc5d3bd14…` |
| `A+R2` | *(free)* | — | — |
| `B+R2` | *(free)* | — | — |
| `X+R2` | `gear_sonic/data/motions/x2_recorded/demo_gestures/chicken_001.pkl` | 34.7 KB | 9.7 s | `5ef2baf6710e…` |
| `Y+R2` | `gear_sonic/data/motions/x2_recorded/demo_gestures/right_wave_001.pkl` | 15.5 KB | 6.6 s | `525c50b12dda…` |

## E-stop acknowledgment

The `L1+R1+L2+R2` chord publishes a stop and then plays this PKL as a visible acknowledgment gesture (set `ESTOP_FOLLOWUP_PKL = None` in the launcher to skip).

| chord | PKL | size | duration | sha-256 |
|-------|-----|-----:|---------:|---------|
| `L1+R1+L2+R2 ack` | `gear_sonic/data/motions/x2_recorded/mc_gestures/shake_head_001.pkl` | 3.5 KB | 5.7 s | `f796b5614283…` |

## Other bundled files

| category | path | size |
|----------|------|-----:|
| playlist | `gear_sonic/data/motions/playlists/relaxed_walk_forward_v1.yaml` | 2.7 KB |
| playlist | `gear_sonic/data/motions/playlists/relaxed_walk_one_left_turn_v1.yaml` | 2.2 KB |
| playlist | `gear_sonic/data/motions/playlists/relaxed_walk_one_right_turn_v1.yaml` | 2.2 KB |
| playlist | `gear_sonic/data/motions/playlists/relaxed_walk_two_right_turns_v1.yaml` | 2.6 KB |
| script | `gear_sonic/scripts/play_xbox_controller.py` | 40.8 KB |
| doc | `xbox_controller_commands.md` | 10.4 KB |

## Regenerating the bundle

Re-run the builder whenever the binding map changes:

```bash
.venv/bin/python -m gear_sonic.scripts.build_demo_bundle
```

Pass `--output <dir>` to stage somewhere other than `./demo_bundle/`, `--no-playlists` / `--no-script` / `--no-cheatsheet` to slim the bundle, or `--tar` to also emit a gzipped tarball next to the bundle directory.
