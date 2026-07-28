# Multi-node elastic SONIC training on a Nebius GPU cluster

How to run one SONIC training job across multiple preemptible GPU nodes with
automatic recovery when nodes die and rejoin. Everything here was built and
**verified live 2026-07-28** on the 4-node H100 cluster (X2 dance-finetune v5
resume, 32 GPUs), including a deliberate kill-a-node drill in both directions
(32→24 ranks on node loss, 24→32 on rejoin).

## Architecture

```
gpu-cluster-node-reserved-main   RESERVED (never preempted)
  ├─ c10d rendezvous store  <reserved-node-ip>:29400   (RDZV_IS_HOST=1)
  ├─ cluster_watchdog.sh    restarts STOPPED workers via nebius CLI
  └─ elastic supervisor     (same script as workers)
gpu-cluster-node-1/2/3[...]      PREEMPTIBLE workers
  └─ elastic supervisor     joins rendezvous; systemd unit auto-joins on boot
/mnt/shared                      shared filesystem (virtiofs), all nodes
  ├─ ckpts/<run>/last.pt         checkpoint every 50 iters + meta.yaml (wandb id)
  └─ corpus/*.pkl                motion corpora (absolute paths in overrides)
```

Every node runs the SAME launcher: `torchrun --nnodes=MIN:MAX --rdzv-backend=c10d
--rdzv-endpoint=<reserved-ip>:29400` wrapped in a supervisor loop
(`gear_sonic/scripts/cloud/run_elastic_multinode.sh`). Accelerate picks up
torchrun's env (RANK/WORLD_SIZE/LOCAL_RANK) — no `accelerate launch` in
multi-node mode.

## Starting a new run

1. Put the corpus pkl(s) on `/mnt/shared/corpus/` (absolute-path overrides are
   proven to work; the loaders don't care).
2. Pick a **fresh** `EXPERIMENT_DIR` on `/mnt/shared/ckpts/` — a stale
   `meta.yaml` in a reused dir re-attaches the old wandb run.
   To *continue* a previous run instead, seed the dir with its `last.pt` (+
   `meta.yaml` for wandb continuity).
3. Write a per-run launcher (copy `launch_x2_elastic.sh` pattern) exporting:
   `EXP_NAME`, `NUM_ENVS` (per GPU), `NUM_ITERS`, `MOTION_FILE`,
   `EXPERIMENT_DIR`, `EXTRA_FLAGS`; copy it to every node as
   `~/launch_elastic_current.sh` (the systemd unit runs that name on boot).
4. Launch: on the reserved node `RDZV_IS_HOST=1 bash launcher` (tmux), plain
   `bash launcher` on workers, or just `sudo systemctl start elastic-supervisor`
   everywhere.

## Semantics you must know

- **`num_learning_iterations` is ABSOLUTE**, not relative: it's the total-step
  target. Resuming a 2500-step checkpoint with `NUM_ITERS=500` exits
  immediately with rc=0 ("training finished cleanly", zero iterations run).
  Use the final target (e.g. 3000).
- **`+resume=True ++experiment_dir=<dir>`** loads `<dir>/last.pt`, restores
  global_step/optimizer/adaptive-sampling, and re-attaches the wandb run id
  from `<dir>/meta.yaml`. The supervisor adds `+resume=True` automatically
  whenever `last.pt` exists.
- **Any membership change = full restart from last.pt** (node loss AND node
  join). Synchronous NCCL all-reduce has fixed membership; there is no
  shrink-in-place. Cost per event ≈ detection + Isaac boot (~7 min) + redo
  since checkpoint (≤50 iters). torchft-style per-step fault tolerance is the
  upgrade path if preemptions become frequent.
- **Node join while world < MAX** → agents SIGTERM (then SIGKILL — Isaac
  ignores SIGTERM) their workers and re-rendezvous including the newcomer.
  While world == MAX, extra joiners park as hot spares.
- Widening `--nnodes` MIN:MAX (e.g. 3:5) beats idle hot spares economically:
  all nodes train; a loss just shrinks the world.

## Verifying it's ONE job on all GPUs (not N duplicates)

- Read the env of any worker: `tr '\0' '\n' < /proc/<pid>/environ | grep -E
  'RANK|WORLD_SIZE'` → every node must show the same `WORLD_SIZE` and
  **distinct** `GROUP_RANK`/`RANK` ranges (0-7, 8-15, ...). Duplicates would
  all show WORLD_SIZE=8, GROUP_RANK=0.
- wandb `episode` counter increments by `num_envs × world_size` per iteration
  (e.g. 393,216 at 32×12288) — the slope kinks visibly when world size
  changes. `tot_timesteps` = 24× that.
- Exactly ONE wandb run: only global rank 0 calls wandb.init. **wandb's
  system/GPU panel therefore shows only rank 0's node** — that's expected,
  not a bug; training metrics are still aggregated across all ranks.
- Console iteration blocks print on **global rank 0's node**, which is
  whichever node grabbed rank 0 at rendezvous — NOT necessarily the reserved
  node. Grep all nodes' `~/elastic.log` for "Iteration time".

## Health monitoring — the honest signals

- **Frozen iteration count** is the stall signal. GPU util is a LIE during
  stalls: ranks blocked in a dead collective spin at 100%.
- Observed in the drill: the NCCL/process-group timeout
  (`SONIC_PG_TIMEOUT_S`, default 600s here) did NOT reliably abort the stuck
  collective (in-flight `ncclCommAbort` hang — known NCCL failure mode). The
  working remediation: **kill the stuck worker processes** (`train_agent_trl`)
  on surviving nodes; agents/supervisors handle the rest automatically. A
  stall-detector loop (iteration count unchanged for ~5 min → kill local
  workers) is the robust automation; monitor scripts already track the count.
- Scaling reference (H100 + 8×400G IB, 12288 envs/GPU, X2 v5): 1 node ≈ 8.0s/iter,
  3 nodes ≈ 8.9s, 4 nodes ≈ 9.25s → 3.5× env throughput at 4 nodes (~86%
  efficiency). Watch envs/sec, not iteration time.

## Fabric (Nebius InfiniBand — measured on this cluster 2026-07-28)

Each node has **8 InfiniBand rails: mlx5 HCAs, one per GPU, `ibstat` Rate
400 Gb/s (NDR) each, all PORT_ACTIVE** → 3.2 Tb/s aggregate per node. This
is Nebius's GPU-cluster fabric — you get it by creating the instances inside
a "GPU cluster" object; standalone instances get no IB. The rail-per-GPU
design lets NCCL do GPUDirect RDMA: each GPU's gradient shard leaves its own
NIC without bouncing through host memory.

- `eth0` (VPC network) is management-only: ssh, the c10d rendezvous TCP
  store, wandb, and the shared virtiofs filesystem. Its speed is irrelevant
  to the all-reduce (gradients never touch it when IB is selected).
- The `ibp*` IPoIB interfaces show DOWN — normal. NCCL uses IB **verbs**
  directly, not IP-over-IB.
- **Verify NCCL actually picked IB** (it auto-selects but can silently fall
  back to `NET/Socket` over eth0): the launcher sets `NCCL_DEBUG=INFO`,
  `NCCL_DEBUG_SUBSYS=INIT,NET`; grep an elastic.log for
  "NCCL INFO Using network IB" (bad: "Using network Socket").
- Sizing intuition: the per-iteration all-reduce moves ~2× model size
  (~400 MB checkpoint → sub-second on even one 400G rail), so the observed
  ~1s/iter multi-node premium is dominated by synchronization latency and
  rank skew (slowest Isaac step sets the pace), NOT bandwidth. Don't buy
  bandwidth to fix it; reduce per-rank jitter instead.

## Auto-heal (no human in the loop)

Nebius does NOT auto-restart preempted instances. Two pieces close the loop:

1. `cluster_watchdog.sh` on the reserved node (tmux `watchdog`): polls
   `nebius compute instance list` every 2 min; any `gpu-cluster-node-*` in
   STOPPED state gets `instance start` retries until capacity returns.
   Needs `~/.nebius/{config,credentials}.yaml` + CLI on the reserved node.
2. `elastic-supervisor.service` (systemd, enabled on every node): runs
   `~/launch_elastic_current.sh` at boot → node rejoins the rendezvous with
   zero manual steps. Reserved node's launcher copy exports `RDZV_IS_HOST=1`.

Full recovery chain (verified except the Nebius-start leg): preemption →
watchdog starts instance → boot → systemd supervisor → rendezvous join →
cluster restarts from last.pt with the node back in.

## Gotchas hit during bring-up (don't rediscover these)

- **`--rdzv-conf is_host=1` is mandatory on the endpoint node.** torchrun's
  c10d host auto-detection resolves the machine hostname (127.0.1.1 on cloud
  images) and never matches the endpoint IP → NO node binds the store → every
  node times out as a client ("client socket has timed out ... <reserved-node-ip>,
  29400"). TCPStore clients retry connection-refused until the deadline, so
  the failure looks like a timeout, not a refusal.
- Private IPs everywhere for training; public IPs are only for ssh and can be
  dropped (quota) — reach IP-less nodes via the reserved node as jump host.
- **The rendezvous store lives INSIDE the host node's torchrun process.** If a
  membership-change restart makes that torchrun exit (rather than restart
  in-place), the store dies too; every other node gets
  "[c10d] waitForInput ... timed out" and its supervisor re-arms. Self-healing
  (the host's next torchrun rebinds the store, everyone re-joins) but it costs
  one extra restart round. Observed during the rejoin half of the drill.
- Checkpoint cadence is `save_last_frequency=50` in ModelSaveCallback — fine
  as-is; last.pt lands on the shared FS so any node can resume it.
- Isaac boot ≈ 7 min of the recovery time; corpus read from the shared FS is
  seconds (don't blame virtiofs I/O).
- `conda activate env_isaaclab` may resolve the wrong python in odd shells —
  invoke `~/miniconda3/envs/env_isaaclab/bin/python` (or let the launcher
  `conda activate` in a fresh login shell).

## File map

- `gear_sonic/scripts/cloud/run_elastic_multinode.sh` — elastic supervisor (any robot/exp)
- `gear_sonic/scripts/cloud/cluster_watchdog.sh` — reserved-node auto-restart loop
- `gear_sonic/scripts/cloud/elastic-supervisor.service` — boot-time auto-join unit
- `gear_sonic/scripts/cloud/bootstrap_fresh_node.sh` — new-node setup (conda,
  IsaacLab, repo); then: creds, `/mnt/shared` symlink, launcher copy, enable unit
