#!/usr/bin/env python3
"""x2_pc3_render.py -- render a text panel onto the X2's face display.

Runs ON PC3 (the RK3588S interaction unit). Reads a JSON spec on stdin, renders
an 800x480 card with PIL, encodes it with the hardware h264 encoder, and posts it
to the flutter-pi face UI over its local HTTP API.

Rendering happens here rather than on PC2 so each update ships ~200 bytes of JSON
instead of a ~100KB video over the robot's internal LAN.

  echo '{"title":"X2","lines":[["INFO","planner up"]]}' | ./x2_pc3_render.py

Spec fields (all optional except lines):
  title   str            header text
  lines   [[lvl, msg]]   log rows, oldest first; lvl one of
                         INFO/WARN/ERR/OK/TTS/>>>  (drives the row colour)
  accent  "#RRGGBB"      header bar colour
  banner  str            single large centred line INSTEAD of the log rows
  sub     str            smaller line under the banner
  out     str            mp4 path to write (default /tmp/x2_panel.mp4)
  post    bool           push to the face UI (default true)

Notes:
  - ffmpeg on PC3 has NO drawtext filter, so text is rasterised with PIL.
  - Only h264_v4l2m2m / h264_rkmpp are available (no libx264).
  - The face UI enforces priority: a request BELOW the currently playing
    priority is silently rejected (code 1100). TTS playback raises it to 60,
    so panels post at 100.
"""
import json
import subprocess
import sys
import urllib.error
import urllib.request

W, H = 800, 480
FACE_UI = "http://127.0.0.1:18080"
PRIORITY = 100                     # must exceed the TTS talking animation (60)
MODE_LOOP = 2                      # 1 = play once, 2 = loop

FONT_DIR = "/usr/share/fonts/truetype/liberation"
BG = (8, 10, 20)
LEVEL_COLOURS = {
    "INFO": (150, 200, 255),
    "OK":   (120, 255, 170),
    "WARN": (255, 205, 110),
    "ERR":  (255, 120, 120),
    "TTS":  (255, 210, 120),
    ">>>":  (120, 255, 170),
}


def _font(name, size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(f"{FONT_DIR}/{name}", size)
    except OSError:
        return ImageFont.load_default()


def render(spec, png_path):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    accent = spec.get("accent") or "#12245a"
    r, g, b = (int(accent.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    d.rectangle([0, 0, W, 58], fill=(r, g, b))
    d.text((22, 13), spec.get("title", "X2"), font=_font("LiberationSans-Bold.ttf", 30),
           fill=(255, 255, 255))
    d.ellipse([W - 46, 22, W - 30, 38], fill=(90, 220, 140))

    banner = spec.get("banner")
    if banner:
        # Single large message (greetings, alerts) instead of the log rows.
        fb = _font("LiberationSans-Bold.ttf", 56)
        d.text(((W - d.textlength(banner, font=fb)) / 2, 170), banner, font=fb,
               fill=(255, 255, 255))
        sub = spec.get("sub")
        if sub:
            fs = _font("LiberationSans-Bold.ttf", 34)
            d.text(((W - d.textlength(sub, font=fs)) / 2, 250), sub, font=fs,
                   fill=(120, 215, 255))
    else:
        fm = _font("LiberationMono-Regular.ttf", 19)
        fmb = _font("LiberationMono-Bold.ttf", 20)
        y = 82
        # Newest rows sit at the bottom; keep only what fits on the panel.
        for lvl, msg in spec.get("lines", [])[-9:]:
            colour = LEVEL_COLOURS.get(lvl.upper(), (185, 195, 215))
            d.text((26, y), lvl.upper()[:4].ljust(4), font=fmb, fill=colour)
            d.text((96, y), str(msg)[:74], font=fm, fill=colour)
            y += 40

    img.save(png_path)


def encode(png_path, mp4_path):
    for enc in ("h264_rkmpp", "h264_v4l2m2m"):
        p = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-loop", "1", "-i", png_path,
             "-c:v", enc, "-pix_fmt", "yuv420p", "-r", "25", "-t", "2", "-y", mp4_path],
            capture_output=True, text=True)
        if p.returncode == 0:
            return enc
    raise RuntimeError(f"no working h264 encoder: {p.stderr.strip()[:200]}")


def post(mp4_path):
    body = json.dumps({"video_path": mp4_path, "mode": MODE_LOOP,
                       "priority": PRIORITY}).encode()
    req = urllib.request.Request(f"{FACE_UI}/PlayVideo", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def main():
    spec = json.load(sys.stdin)
    mp4 = spec.get("out", "/tmp/x2_panel.mp4")
    png = mp4.rsplit(".", 1)[0] + ".png"
    render(spec, png)
    encode(png, mp4)
    if spec.get("post", True):
        resp = post(mp4)
        # code 1100 == rejected because something higher-priority is playing.
        if not resp.get("success"):
            print(f"face_ui rejected: {resp.get('message')}", file=sys.stderr)
            sys.exit(3)
    print("ok")


if __name__ == "__main__":
    main()
