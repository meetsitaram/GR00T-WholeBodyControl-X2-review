"""Subscribe to and pretty-print the C++ deploy's ``x2_debug`` ZMQ stream.

Companion to :mod:`mock_vla_publish_stand_token` — together they form the M2
acceptance gate that proves the X2 deploy harness's post-VLA path works end
to end without a real GR00T model.

What this script does
---------------------

1. Connects a SUB socket to ``tcp://<host>:<port>`` (default 5557, which is
   the X2 deploy's ``x2_debug`` output port — see
   ``docs/source/references/x2_zmq_protocol.md``).
2. Filters on the configured topic prefix (``x2_debug`` by default).
3. Decodes each message with
   :func:`gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder.unpack_message`.
4. Prints (or summarises) each frame, with a final report on the joint-pose
   drift relative to the trained default standing pose so the operator can
   eyeball whether the C++ deploy + sim survived the mock-VLA run.

Acceptance gate
---------------

When invoked alongside :mod:`mock_vla_publish_stand_token` and
``deploy_x2.sh sim --input-type zmq``, this script must report:

* ``frames_received >= 0.9 * (rate * duration)`` (no message starvation),
* ``max abs(body_q - default_pose) < 0.05 rad`` (robot stayed standing),
* No ``policy_safety_event`` (the deploy did not trip its tilt watchdog).

Usage
-----

::

    .venv/bin/python gear_sonic/scripts/dump_x2_debug.py \\
        --port 5557 --topic x2_debug --duration 10
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    DecodedMessage,
    HEADER_SIZE,
    unpack_message,
)


@dataclass
class RunSummary:
    """Aggregated stats over a dump_x2_debug run.

    Used both for live printing and for JSON export when ``--json-out`` is
    set; the latter lets the M2 acceptance gate be scripted (e.g. in a CI
    job) instead of eyeballed.
    """

    frames_received: int = 0
    duration_s: float = 0.0
    keys_seen: list[str] = field(default_factory=list)
    max_token_norm: float = 0.0
    max_body_q_drift: float | None = None  # filled in when default_pose is known
    safety_events: int = 0
    drops_estimated: int = 0  # set when --rate is provided

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["keys_seen"] = sorted(set(self.keys_seen))
        return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument(
        "--port",
        type=int,
        default=5557,
        help="ZMQ SUB port. Must match the deploy's --debug-port.",
    )
    parser.add_argument("--topic", default="x2_debug")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Total subscribe time (0 = until Ctrl-C).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help=(
            "Expected publish rate (Hz) used to estimate dropped frames in the "
            "summary. 0 = skip the drop estimate."
        ),
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print every Nth frame to stdout. Set to 1 to print every frame.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Path to write the run summary JSON. Used by automated gates.",
    )
    parser.add_argument(
        "--default-pose",
        type=str,
        default=None,
        help=(
            "Optional path to a .npy file containing the trained default body "
            "pose (shape (NUM_DOFS,) float32) for drift checking."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-frame prints (still emits the final summary).",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help=(
            "Optional path to write a per-frame CSV trace of selected scalar / "
            "low-D fields (control_tick, ramp_alpha, body_q_*, body_dq_*, "
            "left_hand_q_*, right_hand_q_*, base_quat_*, last_action_*). One "
            "row per received x2_debug frame; useful for plotting the actual "
            "joint motion when debugging a 'frozen' policy."
        ),
    )
    return parser.parse_args(argv)


def _summarize_message(msg: DecodedMessage, prefix: str = "") -> str:
    parts = []
    for name in sorted(msg.fields):
        arr = msg.fields[name]
        # Compact representation: shape + min/max for numeric arrays, value
        # for length-1 arrays.
        if arr.size == 1:
            try:
                value = arr.item()
                parts.append(f"{name}={value!r}")
            except Exception:
                parts.append(f"{name}=<{arr.shape}{arr.dtype}>")
        elif np.issubdtype(arr.dtype, np.number):
            parts.append(
                f"{name}<{arr.shape}{arr.dtype}>="
                f"[{float(arr.min()):+.3f}..{float(arr.max()):+.3f}]"
            )
        else:
            parts.append(f"{name}<{arr.shape}{arr.dtype}>")
    return f"{prefix}v={msg.version} count={msg.count} | " + "  ".join(parts)


def _maybe_drift(
    fields: dict[str, np.ndarray], default_pose: np.ndarray | None
) -> float | None:
    if default_pose is None:
        return None
    body_q = fields.get("body_q") or fields.get("body_q_measured")
    if body_q is None:
        return None
    body_q_arr = np.asarray(body_q).reshape(-1)
    if body_q_arr.shape != default_pose.shape:
        return None
    return float(np.max(np.abs(body_q_arr - default_pose)))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    default_pose: np.ndarray | None = None
    if args.default_pose:
        default_pose = np.load(args.default_pose).astype(np.float64, copy=False).reshape(-1)
        print(
            f"[dump_x2_debug] default_pose loaded from {args.default_pose} "
            f"(shape={default_pose.shape})",
            flush=True,
        )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    sock.setsockopt(zmq.RCVHWM, 10)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{args.host}:{args.port}")
    print(
        f"[dump_x2_debug] SUB connected to tcp://{args.host}:{args.port} "
        f"(topic={args.topic!r})",
        flush=True,
    )

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    summary = RunSummary()
    start = time.monotonic()
    deadline = float("inf") if args.duration <= 0.0 else start + args.duration
    seen_keys: set[str] = set()

    # Lazy-opened CSV writer. We don't know which fields the deploy will
    # publish until the first frame arrives, so we wait until we've seen
    # one and then commit to that header for the rest of the run.
    csv_file = None
    csv_writer = None
    csv_fields: list[str] | None = None

    def _flatten_for_csv(fields: dict[str, Any]) -> dict[str, Any]:
        """Expand 1-D arrays into per-element columns; pass scalars through.

        Anything 2-D or higher is summarised as its L2 norm so the CSV
        stays human-readable. Use ``--json-out`` if you need the full
        tensors.
        """
        flat: dict[str, Any] = {}
        for k, v in fields.items():
            arr = np.asarray(v)
            if arr.ndim == 0:
                flat[k] = float(arr)
            elif arr.ndim == 1:
                if arr.size == 1:
                    flat[k] = float(arr[0])
                else:
                    for i, x in enumerate(arr.tolist()):
                        flat[f"{k}_{i}"] = float(x)
            else:
                flat[f"{k}_l2"] = float(np.linalg.norm(arr))
        return flat

    stop_requested = {"flag": False}

    def _on_signal(signum, _frame):  # type: ignore[unused-argument]
        print(f"[dump_x2_debug] caught signal {signum}, shutting down…", flush=True)
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while not stop_requested["flag"] and time.monotonic() < deadline:
            events = dict(poller.poll(200))
            if sock not in events:
                continue
            raw = sock.recv()
            try:
                msg = unpack_message(raw, expected_topic=args.topic)
            except ValueError as exc:
                print(f"[dump_x2_debug] decode error: {exc}", flush=True)
                continue

            summary.frames_received += 1
            seen_keys.update(msg.fields.keys())

            if args.csv_out is not None:
                row = {"recv_t_mono_s": time.monotonic() - start}
                row.update(_flatten_for_csv(msg.fields))
                if csv_writer is None:
                    import csv as _csv

                    csv_fields = ["recv_t_mono_s"] + sorted(
                        k for k in row.keys() if k != "recv_t_mono_s"
                    )
                    csv_file = open(args.csv_out, "w", newline="")  # noqa: SIM115
                    csv_writer = _csv.DictWriter(csv_file, fieldnames=csv_fields)
                    csv_writer.writeheader()
                # Drop unknown extra keys; they'd raise on a fixed-header
                # writer. The first-frame header is canonical.
                csv_writer.writerow({k: row.get(k, "") for k in csv_fields})

            if "motion_token" in msg.fields:
                summary.max_token_norm = max(
                    summary.max_token_norm,
                    float(np.linalg.norm(msg.fields["motion_token"])),
                )
            drift = _maybe_drift(msg.fields, default_pose)
            if drift is not None:
                summary.max_body_q_drift = (
                    drift
                    if summary.max_body_q_drift is None
                    else max(summary.max_body_q_drift, drift)
                )
            if msg.fields.get("policy_safety_event") is not None and bool(
                np.any(msg.fields["policy_safety_event"])
            ):
                summary.safety_events += 1

            if not args.quiet and (
                summary.frames_received == 1
                or summary.frames_received % max(args.print_every, 1) == 0
            ):
                print(
                    _summarize_message(
                        msg, prefix=f"[dump_x2_debug] #{summary.frames_received:06d}  "
                    ),
                    flush=True,
                )
    finally:
        sock.close(linger=0)
        ctx.term()
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()
            print(f"[dump_x2_debug] wrote {args.csv_out}", flush=True)
        summary.duration_s = time.monotonic() - start
        summary.keys_seen = sorted(seen_keys)
        if args.rate > 0.0 and summary.duration_s > 0.0:
            expected = int(args.rate * summary.duration_s)
            summary.drops_estimated = max(0, expected - summary.frames_received)

        report = summary.to_dict()
        print("[dump_x2_debug] run summary:", flush=True)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True))
            print(f"[dump_x2_debug] wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
