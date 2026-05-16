# X2 motor-state monitoring (PC2 daemon)

`gear_sonic_deploy/scripts/x2_motor_monitor.py` is the third tmux
session in the split-topology deploy (alongside the C++ deploy and
the hand bridge). It runs continuously on PC2, subscribes to the
robot's `JointStateArray` and `JointCommandArray` topics, polls the
MC action mode at 1 Hz, and produces:

1. A daily-rotating JSONL log on PC2
   (`/var/log/x2/motor_monitor.<YYYY-MM-DD>.jsonl` by default).
2. A compact ZMQ summary on `tcp://0.0.0.0:5567` topic
   `motor_monitor` that the laptop's `quest3_manager_x2.py`
   subscribes to and appends to `manager_sidecar.jsonl`. So every
   forensic event lands on both machines: the rich one on PC2, the
   cross-stack-mergable one on the laptop.

The daemon is **strictly read-only**. It never publishes a joint
command, it never calls `SetMcAction`, and it never tries to recover
the robot. The deploy's `PoseRefStarvationWatchdog` + `SAFE_IDLE`
state owns the real-time safety path; the monitor's job is purely
observability so a postmortem can answer "why did the robot do
that".

## JSONL record kinds

```
boot         -- one record per process start; lists thresholds + ports
sample       -- one record per --summary-rate-hz tick (default 1 Hz)
event        -- one record per rising-edge condition (see below)
shutdown     -- one record at clean exit
```

`sample` records carry per-group aggregates (max abs tracking error,
max abs velocity, max abs effort, max KP) and the top-5 joints by
absolute tracking error. They include the latest MC action mode and
status so a `tail -f` of the JSONL gives a continuously-updating
snapshot of MC state.

`event` records carry the joint name + numerical signal that tripped
them. Five trip kinds are currently emitted:

| Type | Trip condition | Default threshold |
|---|---|---|
| `mc_mode_change` | Latest `GetMcAction` differs from previous | (always emit on edge) |
| `tracking_error_spike` | `\|target - position\|` exceeds threshold on any joint | 0.30 rad (~17 deg) |
| `limit_proximity` | Commanded target is within margin of either soft limit | 0.05 rad (~3 deg) |
| `state_staleness` | No `JointStateArray` received for joint within window | 0.5 s |
| `command_staleness` | No `JointCommandArray` received for joint within window | 1.0 s |

All thresholds are `--tracking-error-warn-rad`, `--limit-margin-rad`,
`--stale-state-s`, `--stale-command-s` flags on the daemon CLI.
Hysteresis: an event re-arms only after the signal drops to <= half
the trip threshold, so a flapping joint doesn't spam the JSONL.

## Reading the JSONL

For interactive analysis, the easiest approach is `jq`:

```bash
# All events from the last hour:
jq -c 'select(.kind == "event")' /var/log/x2/motor_monitor.2026-05-15.jsonl

# Just MC mode transitions:
jq -c 'select(.type == "mc_mode_change")' /var/log/x2/motor_monitor*.jsonl

# Top 5 joints by max tracking error in the last 60 samples:
jq -c 'select(.kind == "sample") | .top_tracking_err' \
    /var/log/x2/motor_monitor.2026-05-15.jsonl | tail -60
```

For postmortem-style cross-stream analysis, use the
`x2_freeze_postmortem.py` tool which aligns this JSONL with the
deploy's per-tick CSVs and the manager sidecar JSONL on a single
wall-clock timeline:

```bash
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh postmortem \
    --center-ts "2026-05-15T19:24:30" --window-s 30
```

The wrapper rsyncs the PC2 logs back to the laptop and then runs the
analysis tool with the matching `manager_sidecar.jsonl` from the
laptop's most recent `/tmp/x2_quest3_planner_stack-*` log dir.

## ZMQ summary wire format

The motor monitor PUBs every cycle (default 1 Hz) on
`tcp://0.0.0.0:5567` with topic `motor_monitor`:

```
frame 0:  b"motor_monitor"
frame 1:  utf-8 JSON with shape:
          {
            "sample": <sample record as in JSONL>,
            "events": [<zero or more event records>]
          }
```

The manager's `_motor_monitor_loop` background thread decodes each
multipart message and appends the JSON payload to the manager's
sidecar JSONL via `_sidecar_write` (which is thread-safe via
`_sidecar_lock`).

## Soft-limit override

The default soft-limit table inside the daemon is conservative (see
`DEFAULT_SOFT_LIMITS` in `x2_motor_monitor.py`); the values were
pulled from the X2 URDF + sim configs and are intentionally generous
on the wrists. To tighten or loosen specific joints, write a JSON
file mapping name → [low, high] in radians, and pass
`--soft-limits-json /path/to/file.json`. Missing entries fall back to
the defaults; explicit entries override.

Example:

```json
{
  "left_knee_joint":  [-0.05, 2.50],
  "right_knee_joint": [-0.05, 2.50]
}
```

## Behavioural defaults

* Subscribe QoS matches MC's HAL: BEST_EFFORT, KEEP_LAST(1), VOLATILE.
  This is the only QoS profile that reliably sees MC's per-tick
  publishers (verified via `x2_scan_mc_motors.py probe_publishers`).
* Per-joint rolling buffers hold 60 s @ 50 Hz = 3000 samples. The
  per-cycle aggregate doesn't use the full buffer, but it's there so
  a follow-up FFT-based oscillation detector can be bolted on without
  having to re-collect the data (see TODO at the bottom of the
  daemon source).
* Daily file rotation is on by default (`--no-rotate` to disable).
  The monitor opens a new `<base>.<YYYY-MM-DD>.jsonl` whenever the
  date rolls over while the daemon is running, no signal needed.
