# `pc2_kplanner_onnx.py` changes

Three edits. Applied against the `--cmd-bind` build.

---

## 1. Module-level guard state and ZMQ subscriber

Inserted immediately before `def intent_to_velocity(`.

```python
# ---- forward-obstacle guard -------------------------------------------------
# Subscribes to scan_guard_pub.py over ZMQ rather than importing rclpy: this
# process runs in the gear_sonic venv, which has no ROS bindings, and the gait
# loop is the last place to add an import that can fail.
#
# Clamped HERE (planner_cmd ingest) rather than in the pad bridge so every
# input source is covered -- pad, VR, anything else that publishes to this
# socket, which kplanner binds under --cmd-bind.
#
# FAIL-OPEN: stale or absent guard data means no clamping. Freezing a walking
# biped because a sensor process died is worse than not clamping; the operator
# deadman is the real stop.
_GUARD_PORT = 5571
_GUARD_STALE_S = 0.5
_guard_state = {"blocked": False, "dist": float("inf"), "ts": 0.0}
# Latched stop: once an obstacle trips the guard, forward stays dead until the
# operator releases the deadman (all sticks zero). Auto-release on a clear path
# would let the robot resume walking without a human deciding to.
_guard_latched = {"on": False}


def _scan_guard_thread(stop_event) -> None:
    import zmq as _zmq
    ctx = _zmq.Context.instance()
    sock = ctx.socket(_zmq.SUB)
    sock.setsockopt(_zmq.SUBSCRIBE, b"scan_guard")
    sock.setsockopt(_zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://127.0.0.1:{_GUARD_PORT}")
    log.info("scan guard: SUB tcp://127.0.0.1:%d", _GUARD_PORT)
    while not stop_event.is_set():
        try:
            _, payload = sock.recv_multipart()
        except Exception:  # noqa: BLE001 -- timeout is normal
            continue
        try:
            d = json.loads(payload)
            _guard_state["blocked"] = bool(d["blocked"])
            _guard_state["dist"] = float(d["dist"])
            _guard_state["ts"] = time.monotonic()
        except Exception:  # noqa: BLE001
            continue
    sock.close(linger=0)


def _guard_blocked() -> bool:
    if time.monotonic() - _guard_state["ts"] > _GUARD_STALE_S:
        return False
    return _guard_state["blocked"]
```

---

## 2. The clamp, in `_zmq_command_thread`

Right after the three sticks are parsed from the payload.

```python
                stick_fwd = float(payload.get("stick_fwd", 0.0))
                stick_side = float(payload.get("stick_side", 0.0))
                stick_yaw = float(payload.get("stick_yaw", 0.0))
                if _guard_blocked():
                    if not _guard_latched["on"]:
                        log.warning("OBSTACLE %.2fm -> LATCHED; release the "
                                    "deadman to reset", _guard_state["dist"])
                    _guard_latched["on"] = True
                elif (stick_fwd == 0.0 and stick_side == 0.0
                      and stick_yaw == 0.0 and _guard_latched["on"]):
                    # deadman released (bridge sends an all-zero frame) -> reset
                    log.info("guard latch reset")
                    _guard_latched["on"] = False
                if _guard_latched["on"]:
                    stick_fwd = 0.0
                    stick_side = 0.0
```

Yaw is deliberately untouched — zeroing it too would trap the robot facing a
wall with no way to turn away.

And the same latch applied to the direct-velocity path, since VR-style inputs
send `target_velocity` instead of sticks:

```python
                        direct_velocity = tuple(float(v) for v in target_velocity)
                        if _guard_latched["on"] and direct_velocity[0] > 0.0:
                            log.warning("OBSTACLE %.2fm -> direct vx held",
                                        _guard_state["dist"])
                            direct_velocity = (0.0, 0.0,
                                               direct_velocity[2],
                                               direct_velocity[3])
```

---

## 3. Thread startup

Immediately after the existing `cmd-zmq` thread is started and appended.

```python
    # Forward-obstacle guard feed (scan_guard_pub.py over ZMQ). Fails open:
    # if that process is not running, _guard_blocked() stays False and the
    # planner behaves exactly as before.
    thr = threading.Thread(
        target=_scan_guard_thread, args=(stop_event,),
        name="scan-guard", daemon=True,
    )
    thr.start()
    threads.append(thr)
```

---

## Sanity check after editing

This file keeps a 35 kg biped upright. A syntax error means it will not start
at all.

```bash
python3 -c "import ast; ast.parse(open('pc2_kplanner_onnx.py').read()); print('ok')"
grep -n "_guard_blocked\|_scan_guard_thread\|_guard_latched" pc2_kplanner_onnx.py
```
