#!/usr/bin/env python3
"""Terminal monitor for the scan guard's ZMQ feed.

    python3 watch_guard.py

Prints BLOCKED/clear and the nearest forward distance. Throttled to 4 Hz: the
publisher runs at 50 Hz and printing every frame over ssh feels laggy.
"""
import json
import time

import zmq

PORT = 5571

sock = zmq.Context().socket(zmq.SUB)
sock.setsockopt(zmq.SUBSCRIBE, b"scan_guard")
sock.connect(f"tcp://127.0.0.1:{PORT}")
print("watching scan_guard -- ctrl-c to stop")

last = 0.0
try:
    while True:
        _, payload = sock.recv_multipart()
        now = time.time()
        if now - last < 0.25:
            continue
        last = now
        d = json.loads(payload)
        flag = "BLOCKED" if d["blocked"] else "clear  "
        print(f"\r{flag}  dist={d['dist']:6.2f}m   ", end="", flush=True)
except KeyboardInterrupt:
    print()
