# X2 Interaction Layer — speaker and face display

Output-only interaction for the X2: put text, status logs, or video on the
robot's face panel, and play audio through its speaker — from any task running
on PC2, without disturbing the AgiBot vendor stack.

Everything here is **additive**. No vendor process is stopped, no control-path
file is modified, and nothing is required for locomotion or the planner. If this
layer is absent or broken, the robot behaves exactly as it did before.

Verified on hardware 2026-07-17/18, across three robot reboots and a network change.

---

## 1. Where things live

| Component | Runs on | Path |
|---|---|---|
| `x2_interact.py` | PC2 | `/home/run/getsolo/interact/x2_interact.py` |
| `x2_face.sh` | PC2 | `/home/run/getsolo/interact/x2_face.sh` |
| `x2_pc3_render.py` | **PC3** | `/opt/x2_interact/x2_pc3_render.py` |
| audio (wav) | **PC3** | `/opt/x2_interact/audio/` |
| media (mp4) | **PC3** | `/opt/x2_interact/media/` |

Repo sources: `gear_sonic_deploy/scripts/{x2_interact.py,x2_pc3_render.py,x2_face.sh}`.

> **Why `/opt` on PC3 and not the usual `getsolo` tree?**
> PC3's `/home/run` is `drwxr-x---` (owner-only) and we connect as `agi`, so the
> PC2 convention cannot be used there. `/opt/x2_interact/` is the world-readable
> equivalent. On PC2 the `getsolo` convention still applies.

---

## 2. Hardware facts this layer depends on

These were established empirically; they are not guesses, and each one dictates a
design choice.

### Speaker — use the dmix device, never the raw hardware

The vendor `aima-audiohal-app` holds ALSA card 1 (rockchip-es8388). Opening
`default`, `hw:1,0` or `plughw:1,0` returns **"Device or resource busy"**.

The working path is **`aplay -D playback_def`** — a dmix device defined in
`/etc/asound.conf` (`ipc_key 1015008`, `ipc_perm 0666`). audiohal is itself a
dmix client, so our audio **mixes with** the vendor stack rather than fighting
it. Chimes and speech cut *through* background music; nothing has to be killed.

Audio must be **48 kHz** (dmix is locked to `rate 48000`). Pre-resample rather
than letting ALSA convert. `-8 dB` is a good level with PCM at 100%.

### Display — flutter-pi, arbitrated by priority

The 800×480 DSI panel is driven by **flutter-pi** holding DRM master. There is no
X and no Wayland, so `/dev/fb0` cannot be drawn to while the face UI runs.

flutter-pi exposes an HTTP API on **`0.0.0.0:18080`** (not loopback — PC2 can
drive it directly). Endpoints: `/PlayVideo`, `/PlayVideoGroup`, `/PlayEmoji`,
`/PlayEmojiGroup`, `/PlayDefaultEmoji`, `/ShowRGB`, `/status`,
`/rpc/aimdk_msgs/srv/GetEmotionResources`.

**Priority is a hard gate.** A request *below* the currently-playing priority is
silently rejected with `code 1100`. TTS raises the face to 60 while the robot
speaks, so this layer posts at **100**. A caller that posts below 60 will appear
to work and then randomly fail whenever the robot has just spoken.

`GET /status` returns the live `{priority, e_id, e_path}` — use it to confirm a
post landed rather than trusting the HTTP return code. Idle is `priority 0`,
`e_id 10` (`idleCalm1`).

### Encoding constraints on PC3

* ffmpeg has **no `drawtext`** filter and **no libx264**.
* Text must therefore be rasterised with **PIL** (9.0.1, present).
* Encode with the hardware encoder **`h264_rkmpp`** (or `h264_v4l2m2m`).
* Native format: h264 yuv420p 800×480 25 fps.

Rendering happens **on PC3** so each panel update ships ~200 bytes of JSON
instead of a ~100 KB mp4 across the robot LAN.

---

## 3. Python API (PC2)

```python
import sys; sys.path.insert(0, "/home/run/getsolo/interact")
from x2_interact import X2Interact

ui = X2Interact()
ui.log("planner ONNX loaded", "OK")      # append a row to the on-screen log
ui.log("pose stream degraded", "WARN")
ui.banner("hello!", "how are you?")      # one large centred message
ui.panel([("OK", "line one"), ("INFO", "line two")])
ui.say("chime_ok")                       # pre-baked wav from the PC3 audio dir
ui.music("bed.wav", loop=True)           # background bed; mixes under speech
ui.stop_music()
ui.restore_face()                        # hand the display back to the robot
```

Levels: `INFO` `OK` `WARN` `ERR` `TTS` `>>>` (they colour the row).

### Behavioural guarantees

This sits next to a live demo, so it is built never to be the thing that breaks:

* **Nothing blocks.** Every call returns immediately; work happens on one
  background thread. A wedged network or dead PC3 cannot stall the caller.
* **Nothing raises.** Failures increment `ui.failures` and are logged, never
  propagated. Check `ui.healthy` if you care.
* **Updates coalesce.** Rapid `log()` calls collapse into a single render, so a
  chatty loop cannot build a backlog of ~1 s renders.
* **Audio is detached.** `aplay` runs under `setsid` on PC3 with its pid
  recorded, so playback survives the SSH session and stays independently
  stoppable. `stop_music()` kills the process group.

### CLI

```bash
python3 /home/run/getsolo/interact/x2_interact.py --log "planner up" --level OK
python3 /home/run/getsolo/interact/x2_interact.py --banner "hello!" --sub "welcome"
python3 /home/run/getsolo/interact/x2_interact.py --say chime_ok
python3 /home/run/getsolo/interact/x2_interact.py --music bed.wav
python3 /home/run/getsolo/interact/x2_interact.py --stop-music --restore
python3 /home/run/getsolo/interact/x2_interact.py --demo        # exercise everything
```

---

## 4. Face takeover — `x2_face.sh`

Pure `curl`, no Python, no ssh, no state file. Safe to call from a daemon or a
pad chord.

```bash
x2_face.sh on              # loop the partner-logo reel
x2_face.sh off             # hand back to the vendor idle face
x2_face.sh toggle          # flip, based on what is ACTUALLY playing
x2_face.sh status          # "ours: ..." or "vendor: ..."
x2_face.sh media <path>    # loop an arbitrary mp4 (path on PC3)
```

**Nothing is shut down.** The stock face is masked by priority, not killed. If
the script never runs, or dies midway, the robot's own face is unaffected.

`toggle` reads real state from `/status` rather than tracking its own flag —
the vendor changes the face underneath us (TTS raises it to `thinking`), so a
script holding a boolean would drift out of sync and start toggling backwards.

Env overrides: `X2_PC3_HOST`, `X2_FACE_MEDIA`, `X2_FACE_PRIO`.

---

## 5. Ignition hook

`x2_pc2/ritual_start_demo.sh` brings the logo reel up at the end of the demo
ignition chain:

```bash
( /home/run/getsolo/interact/x2_face.sh on >/dev/null 2>&1 \
    && echo "$(date +%F_%T) face: logo reel ON" >> $LOG \
    || echo "$(date +%F_%T) face: logo reel FAILED (ignored)" >> $LOG ) &
```

Four deliberate safety properties: it runs **after the pose-stream gate and after
deploy start** so it cannot delay or block ignition; it is **backgrounded** so a
slow PC3 cannot stall the script; **all failures are swallowed** so it never
changes the exit status; and it **logs either way** to `ritual_fired.log`.

The script executes as user `run`, which is what owns `/home/run`.

---

## 6. Building media

Panel cards and reels are 800×480 h264. Logo cards are rendered with PIL and
encoded per-segment, then concatenated with fades.

```bash
# transcode any source video for the panel (letterbox, never stretch; strip audio)
ffmpeg -i in.mp4 \
  -vf "scale=800:480:force_original_aspect_ratio=decrease,\
pad=800:480:(ow-iw)/2:(oh-ih)/2:color=black,fps=25" \
  -c:v h264_rkmpp -pix_fmt yuv420p -an -y out.mp4

# bake audio for the speaker
ffmpeg -i in.mp3 -ar 48000 -ac 2 -c:a pcm_s16le \
  -af "volume=-8dB,afade=t=in:st=0:d=1.5" -y out.wav
```

Audio is **stripped from video** (`-an`) — the speaker is reserved for the
robot's voice, and narration competing with alerts is unusable.

> **Always view rendered media before shipping it.** Two real bugs were caught
> only by extracting a frame and looking at it: a logo that rendered
> black-on-black (wrongly assumed to need inversion), and demo footage that was
> a *competitor's* robot. Neither is visible from a file listing.

---

## 7. Gotchas

* **`pkill -f <pattern>` can match its own command line** and kill the shell
  issuing it, silently doing nothing. Use `pkill -x aplay`, or the module's
  pidfile-based `stop_music()`.
* **The base64 SSH wrapper consumes remote stdin**, so payloads must be embedded
  in the command rather than piped in.
* **Never stage anything in `/tmp` on either machine** — it is cleared on reboot
  and mid-session. Use `/opt/x2_interact/` (PC3) or `/home/run/getsolo/` (PC2).
* **The logo reel and status panels both post at priority 100** and will fight
  over the screen. Decide which owns the display when idle.
* PC3's `/home/run` is not traversable by `agi`; PC2's is not either. Run as
  `run` (`sudo su run -c ...`) for anything under it.

---

## 8. Not included

This layer is **output only**. There is deliberately no microphone capture, no
wake word, and no speech recognition.

Investigation for a future voice interface established:

* The **ES7210 4-channel mic array (card 2)** is held by a `data_loop_capture`
  thread inside `aima-audiohal-app`. It cannot be opened concurrently — unlike
  playback, there is no shared capture path in use.
* `capture_loopback` (`hw:Loopback,1,2`) carries **digital silence** — it is not
  a mic tap.
* `capture_def` (ES8388, card 1) opens successfully but yields **a click then
  silence** (per-second RMS `[8002, 1, 1, 1, ...]`). It is not a usable mic.

So a voice interface requires stopping `aima-audiohal-app` to free the array —
which is riskier than it sounds, because audiohal is a **dmix client on the
shared playback device this layer depends on**. Verify the speaker still works
immediately after any such change.
