# Sim-to-Real Anchor Archive

The matched real-robot vs MuJoCo recordings, comparison plots, and per-anchor `SUMMARY.md` files for the paper provenance archive live alongside the data, **not** in `docs/`:

→ [`data/sim_to_real_anchors/README.md`](../../../data/sim_to_real_anchors/README.md)

This archive is whitelisted in `.gitignore` (`!data/sim_to_real_anchors/`) so the `.npz` recordings, plots, and summaries travel with the repo as a single provenance unit.

## Quick links

- [Anchor B — `casual_walk_v1` (first powered walk)](../../../data/sim_to_real_anchors/anchor_b_iter22k_casual_walk_v1/SUMMARY.md)
- [Anchor C — `showcase_v1` (stand-in-place upper-body reel, tightest sim-to-real)](../../../data/sim_to_real_anchors/anchor_c_iter22k_showcase_v1/SUMMARY.md)
- [Anchor D — `walk_demo_v6` (turn-walk-turn-walk-return, strongest locomotion)](../../../data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/SUMMARY.md)

## Related docs (still in `docs/`)

- [`sim_to_real_recordings_inventory.md`](sim_to_real_recordings_inventory.md) — recorder npz schema, mapping from on-disk artefacts to deploy commands, history of how each recording was located.
