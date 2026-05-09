"""Quest 3 VR data reader via WebSocket.

Runs a WebSocket server that receives head + controller tracking data
from a WebXR app running in the Quest 3 browser, transforms the poses
into the robot coordinate frame, and exposes them through an API that
mirrors PicoReader so the rest of the pipeline can be device-agnostic.

WebXR coordinate system (right-handed, Y-up):
    X = right, Y = up, Z = backward (toward user)

Robot coordinate system (right-handed, Z-up):
    X = forward, Y = left, Z = up

Position transform:
    robot = WEBXR_TO_ROBOT @ webxr
    where WEBXR_TO_ROBOT swaps axes accordingly.

Quaternion transform:
    R_robot = Q * R_webxr * Q^{-1}   (basis change)
"""

import asyncio
import http.server
import json
import os
import socket as _socket
import ssl
import subprocess
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as sRot

# WebXR -> Robot basis-change matrix
# robot_x (forward) = -webxr_z, robot_y (left) = -webxr_x, robot_z (up) = webxr_y
WEBXR_TO_ROBOT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float64,
)

_Q_ROT = sRot.from_matrix(WEBXR_TO_ROBOT)


def _get_lan_ip() -> str:
    """Return the LAN IP address of this machine (best-effort)."""
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _generate_self_signed_cert(cert_dir: str) -> tuple[str, str]:
    """Generate a self-signed TLS certificate for the WebSocket/HTTPS servers."""
    os.makedirs(cert_dir, exist_ok=True)
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        return cert_file, key_file

    print(f"[Quest3] Generating self-signed certificate in {cert_dir}")
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            key_file,
            "-out",
            cert_file,
            "-days",
            "365",
            "-nodes",
            "-subj",
            "/CN=quest3-teleop",
        ],
        check=True,
        capture_output=True,
    )
    print(f"[Quest3] Certificate generated: {cert_file}")
    return cert_file, key_file


def transform_pose_to_robot(
    pos: list | np.ndarray,
    quat_xyzw: list | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a single pose from WebXR frame to robot frame.

    Args:
        pos: [x, y, z] in WebXR coordinates.
        quat_xyzw: [x, y, z, w] quaternion (WebXR / scipy default order).

    Returns:
        robot_pos: (3,) position in robot frame.
        robot_quat_wxyz: (4,) quaternion [w, x, y, z] in robot frame.
    """
    robot_pos = WEBXR_TO_ROBOT @ np.asarray(pos, dtype=np.float64)
    webxr_rot = sRot.from_quat(quat_xyzw)  # scipy: xyzw
    robot_rot = _Q_ROT * webxr_rot * _Q_ROT.inv()
    robot_quat_wxyz = robot_rot.as_quat(scalar_first=True)
    return robot_pos, robot_quat_wxyz


def compute_3pt_pose_from_quest3(data: dict) -> np.ndarray | None:
    """Convert Quest 3 tracking JSON to a (3, 7) VR 3-point pose array.

    The three points are [left_wrist, right_wrist, neck/head], each stored
    as [x, y, z, qw, qx, qy, qz] relative to an estimated root position
    (floor projection of the headset) in the robot coordinate frame.

    Args:
        data: Dict with keys ``head``, ``left``, ``right``, each containing
              ``position`` [x,y,z] and ``orientation`` [x,y,z,w] in WebXR frame.

    Returns:
        (3, 7) float32 ndarray, or *None* if required fields are missing.
    """
    head = data.get("head")
    left = data.get("left")
    right = data.get("right")

    if not head or not left or not right:
        return None
    for part in (head, left, right):
        if "position" not in part or "orientation" not in part:
            return None

    head_pos, head_quat = transform_pose_to_robot(head["position"], head["orientation"])
    left_pos, left_quat = transform_pose_to_robot(left["position"], left["orientation"])
    right_pos, right_quat = transform_pose_to_robot(right["position"], right["orientation"])

    # Estimate root at floor below headset (identity orientation)
    root_pos = np.array([head_pos[0], head_pos[1], 0.0])

    vr_3pt = np.zeros((3, 7), dtype=np.float32)

    vr_3pt[0, :3] = left_pos - root_pos
    vr_3pt[0, 3:] = left_quat

    vr_3pt[1, :3] = right_pos - root_pos
    vr_3pt[1, 3:] = right_quat

    vr_3pt[2, :3] = head_pos - root_pos
    vr_3pt[2, 3:] = head_quat

    return vr_3pt


class Quest3Reader:
    """Background reader for Quest 3 VR data via WebSocket.

    Starts a WebSocket server (optionally with TLS) that the WebXR app
    running in the Quest 3 browser connects to.  Also starts an HTTP(S)
    server to serve the WebXR client page.
    """

    def __init__(
        self,
        ws_host: str = "0.0.0.0",
        ws_port: int = 8765,
        http_port: int = 8443,
        use_ssl: bool = True,
        quiet_periodic: bool = False,
    ):
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.http_port = http_port
        self.use_ssl = use_ssl
        # When True, suppress the per-100-msg "msgs=N fps=X idle" line.
        # The first-packet snapshot and one-shot XR / controller / hand
        # tracking events still log -- those are diagnostic gold.
        self.quiet_periodic = bool(quiet_periodic)

        self._latest: dict | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._connected = False
        self._fps_ema = 0.0
        self._last_t: float | None = None

        self._ws_thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._cert_dir = os.path.join(os.path.dirname(__file__), "quest3_certs")

    # -- lifecycle ------------------------------------------------------------

    def start(self):
        """Start WebSocket and HTTP servers in background threads."""
        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()
        self._http_thread = threading.Thread(target=self._run_http, daemon=True)
        self._http_thread.start()

    def stop(self):
        self._stop.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._ws_thread:
            self._ws_thread.join(timeout=2.0)

    # -- public getters -------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_latest(self) -> dict | None:
        with self._lock:
            return self._latest

    def get_3pt_pose(self) -> np.ndarray | None:
        sample = self.get_latest()
        if sample is None:
            return None
        return sample.get("vr_3pt_pose")

    def get_controller_inputs(self) -> tuple[float, float, float, float]:
        """Returns (left_trigger, right_trigger, left_grip, right_grip)."""
        sample = self.get_latest()
        if sample is None:
            return 0.0, 0.0, 0.0, 0.0
        buttons = sample.get("buttons", {})
        return (
            float(buttons.get("leftTrigger", 0.0)),
            float(buttons.get("rightTrigger", 0.0)),
            float(buttons.get("leftGrip", 0.0)),
            float(buttons.get("rightGrip", 0.0)),
        )

    def get_controller_axes(self) -> tuple[float, float, float, float]:
        """Returns (lx, ly, rx, ry) joystick values.

        Convention: positive lx = right, positive ly = forward.
        The WebXR app is expected to invert the Y axis before sending.
        """
        sample = self.get_latest()
        if sample is None:
            return 0.0, 0.0, 0.0, 0.0
        axes = sample.get("axes", {})
        return (
            float(axes.get("lx", 0.0)),
            float(axes.get("ly", 0.0)),
            float(axes.get("rx", 0.0)),
            float(axes.get("ry", 0.0)),
        )

    def get_buttons(self) -> tuple[bool, bool, bool, bool]:
        """Returns (a, b, x, y) face-button pressed states."""
        sample = self.get_latest()
        if sample is None:
            return False, False, False, False
        buttons = sample.get("buttons", {})
        return (
            bool(buttons.get("a", False)),
            bool(buttons.get("b", False)),
            bool(buttons.get("x", False)),
            bool(buttons.get("y", False)),
        )

    # -- WebSocket server -----------------------------------------------------

    def _make_ssl_context(self) -> ssl.SSLContext | None:
        if not self.use_ssl:
            return None
        try:
            cert_file, key_file = _generate_self_signed_cert(self._cert_dir)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert_file, key_file)
            return ctx
        except Exception as e:
            print(f"[Quest3Reader] SSL setup failed ({e}), falling back to WS")
            return None

    async def _handle_connection(self, websocket):
        self._connected = True
        addr = getattr(websocket, "remote_address", "unknown")
        print(f"[Quest3Reader] Client connected: {addr}")
        try:
            async for message in websocket:
                if self._stop.is_set():
                    break
                self._process_message(message)
        except Exception as e:
            if not self._stop.is_set():
                print(f"[Quest3Reader] Connection error: {e}")
        finally:
            self._connected = False
            print(f"[Quest3Reader] Client disconnected: {addr}")

    def _process_message(self, raw: str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Handle status messages from the WebXR client
        if data.get("_type") == "status":
            event = data.get("event", "unknown")
            if event == "xr_session_started":
                ref = data.get("ref_space", "?")
                supported = ", ".join(data.get("supported", []))
                unsupported = ", ".join(data.get("unsupported", []))
                n_inputs = data.get("input_sources", 0)
                print(f"[Quest3Reader] XR session started!")
                print(f"[Quest3Reader]   Reference space: {ref}")
                print(f"[Quest3Reader]   Supported spaces: {supported}")
                if unsupported:
                    print(f"[Quest3Reader]   Unsupported spaces: {unsupported}")
                print(f"[Quest3Reader]   Input sources (controllers): {n_inputs}")
                if ref != "local-floor":
                    print(f"[Quest3Reader]   WARNING: 'local-floor' not available. Floor height may be wrong.")
                    print(f"[Quest3Reader]   TIP: Set up Guardian on Quest 3: Settings > Physical Space > Space Setup")
            elif event == "xr_ref_space_failed":
                unsupported = ", ".join(data.get("unsupported", []))
                print(f"[Quest3Reader] ERROR: No XR reference space available!")
                print(f"[Quest3Reader]   Unsupported: {unsupported}")
                print(f"[Quest3Reader]   FIX: Set up Guardian on Quest 3:")
                print(f"[Quest3Reader]     1. Press Meta button > Settings > Physical Space > Space Setup")
                print(f"[Quest3Reader]     2. Choose 'Roomscale' or 'Stationary'")
                print(f"[Quest3Reader]     3. Follow prompts to draw boundary, then reload the page")
            elif event == "input_sources_changed":
                count = data.get("count", 0)
                sources = data.get("sources", [])
                print(f"[Quest3Reader] Input sources changed: {count} detected")
                has_controller = False
                has_hand = False
                for s in sources:
                    stype = s.get("type", "unknown")
                    hand = s.get("handedness", "?")
                    has_gpad = s.get("has_gamepad", False)
                    print(f"[Quest3Reader]   {hand}: {stype} (gamepad={'yes' if has_gpad else 'NO'})")
                    if stype == "controller":
                        has_controller = True
                    elif stype == "hand-tracking":
                        has_hand = True
                if has_hand and not has_controller:
                    print(f"[Quest3Reader]   WARNING: Hand tracking detected but NO controllers!")
                    print(f"[Quest3Reader]   FIX: Pick up the physical Quest 3 controllers.")
                    print(f"[Quest3Reader]   The headset will auto-switch to controller mode.")
                elif has_controller:
                    print(f"[Quest3Reader]   Controllers detected — buttons and joysticks active.")
            else:
                print(f"[Quest3Reader] Status: {data}")
            return

        self._msg_count = getattr(self, "_msg_count", 0) + 1
        self._last_log_count = getattr(self, "_last_log_count", 0)

        # Log first message and then every 100th message
        if self._msg_count == 1:
            print(f"[Quest3Reader] First tracking data received! Keys: {list(data.keys())}")
            btns = data.get("buttons", {})
            axes = data.get("axes", {})
            print(f"[Quest3Reader]   buttons: {btns}")
            print(f"[Quest3Reader]   axes: {axes}")
        elif self._msg_count % 100 == 0 and not self.quiet_periodic:
            btns = data.get("buttons", {})
            axes = data.get("axes", {})
            has_input = any(v for k, v in btns.items() if k in ("a", "b", "x", "y") and v)
            has_trigger = any(v > 0.1 for k, v in btns.items() if k not in ("a", "b", "x", "y"))
            has_stick = any(abs(v) > 0.05 for v in axes.values())
            status_parts = []
            if has_input:
                pressed = [k.upper() for k in ("a", "b", "x", "y") if btns.get(k)]
                status_parts.append(f"btns=[{'+'.join(pressed)}]")
            if has_trigger:
                status_parts.append("triggers=active")
            if has_stick:
                status_parts.append(f"sticks=active")
            status = " | ".join(status_parts) if status_parts else "idle"
            print(f"[Quest3Reader] msgs={self._msg_count} fps={self._fps_ema:.1f} {status}")

        vr_3pt_pose = compute_3pt_pose_from_quest3(data)

        now = time.time()
        dt = (now - self._last_t) if self._last_t else 0.0
        if dt > 0:
            inst_fps = 1.0 / dt
            self._fps_ema = (
                inst_fps if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst_fps
            )
        self._last_t = now

        sample = {
            "vr_3pt_pose": vr_3pt_pose,
            "buttons": data.get("buttons", {}),
            "axes": data.get("axes", {}),
            "timestamp_realtime": now,
            "timestamp_monotonic": time.monotonic(),
            "dt": dt,
            "fps": self._fps_ema,
        }
        with self._lock:
            self._latest = sample

    def _run_ws(self):
        try:
            import websockets
        except ImportError:
            print(
                "[Quest3Reader] ERROR: 'websockets' package not installed. "
                "Run: pip install websockets"
            )
            return

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        ssl_ctx = self._make_ssl_context()
        proto = "wss" if ssl_ctx else "ws"

        async def _serve():
            server = await websockets.serve(
                self._handle_connection,
                self.ws_host,
                self.ws_port,
                ssl=ssl_ctx,
            )
            lan_ip = _get_lan_ip()
            print(f"[Quest3Reader] WebSocket server on {proto}://{lan_ip}:{self.ws_port}")
            try:
                await asyncio.get_event_loop().create_future()
            finally:
                server.close()
                await server.wait_closed()

        try:
            self._loop.run_until_complete(_serve())
        except Exception:
            pass

    # -- HTTP server for the WebXR app ----------------------------------------

    def _run_http(self):
        app_dir = os.path.join(os.path.dirname(__file__), "quest3_webxr_app")
        if not os.path.isdir(app_dir):
            print(f"[Quest3Reader] WebXR app not found at {app_dir}")
            return

        class _Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=app_dir, **kwargs)

            def end_headers(self):
                # Prevent browser caching so code changes take effect immediately
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                super().end_headers()

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer((self.ws_host, self.http_port), _Handler)

        proto = "http"
        if self.use_ssl:
            try:
                cert_file, key_file = _generate_self_signed_cert(self._cert_dir)
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(cert_file, key_file)
                server.socket = ctx.wrap_socket(server.socket, server_side=True)
                proto = "https"
            except Exception as e:
                print(f"[Quest3Reader] HTTPS setup failed ({e}), using HTTP")

        lan_ip = _get_lan_ip()
        print(f"[Quest3Reader] Serving WebXR app at {proto}://{lan_ip}:{self.http_port}")

        while not self._stop.is_set():
            server.timeout = 0.5
            server.handle_request()
