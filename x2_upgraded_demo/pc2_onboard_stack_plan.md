# Plan: Full planner stack ON PC2 + Xbox BT + Quest via robot AP (demo untethering)

Goal: `run_x2_quest3_planner_stack` pipeline running **entirely on the robot's
PC2** — no laptop, no home wifi. Operator inputs: **Xbox controller over
Bluetooth** and/or **Quest 3 over the robot's own wifi (AP mode)**.

## 0. Ground truth (probed 2026-07-16)

**PC2 = NVIDIA Jetson Orin** — ARMv8, 8 cores, 15 GB RAM, integrated Orin GPU
(nvgpu), Ubuntu 22.04, 372 GB free disk. Python 3.10 system, no conda.
- WiFi+BT = **Intel AX210 combo** (WiFi 6E + BT 5.3). **AP mode supported** ✓
- **Bluetooth CONFIRMED**: hci0 present, not rf-blocked, service merely
  `disabled` → `systemctl enable --now bluetooth` and pair. ✓
- Already runs: deploy daemons (sonic ONNX via onnxruntime), hand bridge,
  motor monitor — so **onnxruntime on the Orin GPU is proven** ✓
- Robot-internal wired net `develop0` 10.0.1.41: **PC1=10.0.1.40** (motion
  control), **10.0.1.42 = PC3 (probing)**, sensor net 10.11.1.x, and a
  **cellular modem (wwan0)** — potential remote-ops/telemetry path.
- Xbox controller tooling already exists (play_xbox_controller.py, motion-clip
  wire) — Phase 3 = adapt to locomotion driving, not build from scratch.

**Current topology:** laptop runs quest3_manager (WebXR reader: Quest browser
→ wss :8765 / https :8443), x2_kplanner (torch, CUDA), recorder/mux (pose PUB
:5556) → PC2 deploy SUBs over wifi. Target topology: all of that on PC2,
loopback wires, inputs arriving over the robot's own radios.

**Consequence of ARM:** the laptop's x86 conda envs do NOT transfer. Two
viable strategies for the kplanner's neural models (vqvae/pose/root):
- **Strategy A (preferred): export the planner trio to ONNX**, run with the
  already-proven onnxruntime. No torch on PC2 at all.
- Strategy B (fallback): NVIDIA Jetson PyTorch wheels + install motionbricks.
  Heavier, slower to set up, but zero export work.

## Phase 0 — Complete the inventory (½ day, laptop-driven, no risk)
- [ ] Orin module + JetPack version (`cat /etc/nv_tegra_release`, `jetson_release`)
- [ ] onnxruntime version + providers on PC2 (CUDA/TensorRT EP available?)
- [ ] Bluetooth adapter present? (`bluetoothctl list`, `dmesg | grep -i blue`);
      if absent → USB BT dongle (trivial) or wired/2.4GHz-dongle Xbox pad
- [ ] Wifi concurrent AP+STA support (`iw list` "valid interface combinations")
      — decides whether robot AP and internet access can coexist
- [ ] Free CPU/GPU headroom while deploy daemons run (tegrastats during a
      locomotion session)

## Phase 1 — Kplanner-on-Orin feasibility spike (1-2 days) **[CRITICAL PATH]**

**✅ COMPUTE FEASIBILITY EMPIRICALLY CONFIRMED (2026-07-16 night):**
- trtexec on PC2's Orin, synthetic 152M-param fp16 transformer (heavier than
  the real 137M pose model): **mean 4.9 ms, p99 6.2 ms, 205 qps — PASSED.**
  ~60x headroom vs the 300 ms acceptance; full replan estimate ~100 ms worst
  case vs the 1.4 s budget.
- Deploy's onnxruntime is **CPU-only** (libonnxruntime.so 1.16.3, no CUDA EP)
  -> sonic runs on CPU at 50 Hz today -> **the GPU is 100% free** for the
  planner. Memory: all four models ~1 GB vs 15 GB unified. Engine build
  ~1 min. Remaining Phase-1 work = the torch->ONNX export itself (below).

The only real technical risk. Everything else is plumbing.
- [ ] Write ONNX export for the planner trio (NeuralPlannerCore forward =
      vqvae encode + pose/root autoregressive step). Precedent: the SONIC
      tokenizer/decoder already exports (reexport_x2_g1_onnx) — same
      pattern, parity-gated (dump GT → export → max|onnx−pt| check).
      Complication: the pose/root models are autoregressive with KV-style
      state; may need fixed-window stateless export (re-encode 4-frame
      context per replan — fine at our 1.4 s replan cadence).
- [ ] Benchmark on Orin: replan latency (GPU + CPU fallback), publish-loop
      jitter at 20 ms tick. **Acceptance: replan < 300 ms p95** (budget: we
      replan every ~1.4 s; laptop GPU does 5-15 ms; Orin GPU estimate
      50-150 ms — comfortable; CPU-only likely 0.5-2 s — fails).
- [ ] If A stalls: Strategy B timebox (Jetson torch wheels + motionbricks
      pip install, CPU/GPU smoke of replay_pkl_through_kplanner).

## Phase 2 — Stack decomposition on PC2 (1-2 days)
- [ ] Split the 3000-line laptop launcher: PC2 needs manager (quest reader +
      intent decoder), kplanner (ONNX runner), pose mux/recorder in
      **teleop-only** mode (no dataset writing, no cameras), deploy (already
      resident). Write `x2_pc2_full_stack.sh` (systemd-friendly, tmux panes,
      one command) rather than porting run_x2_quest3_planner_stack wholesale.
- [ ] Loopback topology: pose wire binds 127.0.0.1 on PC2 (the 2026-07-16
      safety fix generalizes: deploy SUBs localhost when co-resident).
- [ ] Headless: no viewers anywhere; log files + status line.
- [ ] Model staging: planner ONNX trio + sonic ONNX + clip library
      (X2-clip.ckpt is torch — export/convert the template library too, or
      bake to .npz for the ONNX runner).
- [ ] Sim-parity gate: same stack on laptop in sim vs on PC2 driving MuJoCo
      remotely — identical Replanning logs for a scripted input sequence.

## Phase 3 — Gamepad over Bluetooth (1 day, parallel with Phase 4)

**UPDATE (user info):** the robot's package box INCLUDED a PlayStation
controller — vendor-intended gamepad support, never yet tried. Prefer the
bundled PS pad over Xbox: Ubuntu 22.04 has in-kernel hid_playstation
(DualSense) / hid_sony (DS4) drivers, and pygame reads it like any joystick
(different button/axis indices than Xbox — mapping table needed). Check
whether the vendor's own MC app auto-binds the pad (conflict risk: make sure
OUR reader owns it during demos). Xbox pad remains the fallback.
- [ ] Enable bluetooth service; pair pad (xpadneo driver if Series X/S pad;
      kernel xpad otherwise). Verify /dev/input/js0 + pygame sees it.
- [ ] Write `xbox_locomotion_publisher.py`: pygame.joystick → the SAME
      planner_cmd vocabulary the quest manager publishes (stick_fwd/side/yaw,
      speed_delta on X/Y buttons — mirroring the VR mapping we just built,
      mode toggles, deadman = hold LB or similar, idle on release).
      Precedent: play_xbox_controller.py already does pygame → motion_clip
      wire; this is the locomotion twin. Reuses all kplanner-side logic
      (setpoint, templates) untouched.
- [ ] Latency check BT→command→replan; failsafe: controller disconnect →
      immediate IDLE command.

### Bundled PS pad — vendor label & our mapping (photo 2026-07-16)
The included pad carries AgiBot's own scheme (targets PC1's MC app):
gestures = Create/Option + L1/R1 + face button (Handshake/Wave/Flying-Kiss/
Salute/Raise-Hand/Fist-Bump/Clap); mode combos (↑+△ supine start-up, L2+X
standing-prep, R2+X stable-standing, ↑+X sitting-prep, L1+R1+Create =
ZERO-TORQUE/E-STOP, L2+R2+Create = damping). Red warning: improper E-stop /
zero-torque can damage the robot.

**OWNERSHIP CAVEAT (must resolve first):** if the pad is BT-paired to the
vendor stack (PC1?), our combos could trigger vendor modes SIMULTANEOUSLY
(e.g. a stray L1+R1+Create = instant zero-torque collapse). Find where it
currently pairs; for our demos pair it to PC2 ONLY and confirm the vendor
listener isn't consuming it. Never reuse the vendor's E-STOP combo for
anything else in our mapping.

**Our mapping (goals: launch sonic / drive via kplanner / dance buttons),
mimicking vendor muscle-memory where safe:**
| input | action | wire |
|---|---|---|
| L-stick | fwd/lateral (continuous) | planner_cmd sticks |
| R-stick X | yaw | planner_cmd |
| D-pad up/down | speed setpoint +/-0.1 (mirrors VR X/Y) | speed_delta |
| Options (hold 1s) | START sonic stack (daemons+engage) | pc2 launcher |
| R2+X | engage locomotion (mimics "stable standing") | state machine |
| ↑+X | disengage to idle (mimics "sitting prep" semantics) | state machine |
| Create + face btns | prelisted DANCE/GESTURE clips (4 slots: X/△/□/○) | motion_clip_cmd (play_gesture path, same wire play_xbox_controller.py uses) |
| Option + ↑/↓/→ | 3 more clip slots | motion_clip_cmd |
| L1+R1+Create | reserved = TRUE E-stop passthrough (vendor semantic kept) | deploy stop |
| pad disconnect / no input 500ms | IDLE command | failsafe |

Dance slots load from a small YAML (clip name -> button), so the prelisted
set is editable without code (reuse gestures_v1.yaml pattern).

## Phase 4 — Quest 3 via robot AP (1 day, parallel with Phase 3)
- [ ] NetworkManager hotspot profile on PC2 (nmcli con add type wifi mode ap
      ssid X2-ROBOT ...), static 10.42.0.1, DHCP for clients. Decide band:
      5 GHz for Quest streaming quality.
- [ ] Quest joins X2-ROBOT; browser → https://10.42.0.1:8443 (WebXR needs
      TLS: regenerate the self-signed cert with the AP IP SAN, accept once
      on the headset).
- [ ] Verify Quest3Reader end-to-end on AP (ws :8765): pose rate, dropout
      behavior when operator walks around the robot.
- [ ] Network profile switching story: demo mode (AP) vs dev mode (home
      wifi) — one script to flip; if AP+STA concurrency works (Phase 0),
      keep both.

## Phase 5 — Demo hardening (1 day + rehearsals)
- [ ] One-command bring-up on PC2 (systemd units or a single tmux script),
      auto-start models preloaded; cold-boot-to-driveable < 2 min.
- [ ] Failure drills: input dropout → IDLE; kplanner crash → auto-restart
      with frozen anchor; deploy watchdog behavior confirmed.
- [ ] Thermal/power: tegrastats under full stack + walking for 15 min
      (Orin + planner inference + policy — watch throttling).
- [ ] E-stop story documented for the operator; the "collapse on stop"
      caveat surfaced in the runbook.
- [ ] Full dress rehearsal on hardware with both inputs.

## Risks (ranked)
1. **Planner ONNX export complexity** (autoregressive state) — mitigation:
   stateless fixed-window export; fallback Strategy B (Jetson torch).
2. **Orin headroom** with deploy + planner + WebXR concurrently — mitigation:
   benchmark early (Phase 1), TensorRT EP if needed, pin deploy to cores.
3. **BT adapter absent / pad driver quirks** — mitigation: USB dongle; wired.
4. **WebXR TLS/AP quirks on Quest** — mitigation: known-good self-signed
   cert flow exists (8443 already in use); test early in Phase 4.
5. **Wifi AP throughput/interference at demo venue** — mitigation: 5 GHz,
   channel scan at venue; Xbox BT is the redundant input if wifi is hostile.

## Sequencing & effort
Phase 0 (½ d) → Phase 1 spike (1-2 d, gating) → Phase 2 (1-2 d)
→ Phases 3+4 in parallel (1 d each) → Phase 5 (1 d + rehearsals).
**~5-7 working days**, with the Phase-1 benchmark as the go/no-go gate:
if Orin can't run the planner acceptably, fallback demo topology = laptop
kplanner + robot AP (laptop joins the robot's wifi — still no venue wifi
dependency, one extra box).

## Open questions for the user
- Which Xbox pad generation (BT protocol / driver path differs)?
- Must Quest and Xbox work in the SAME session (hot-swap) or either-or per
  session? (Hot-swap = mux arbitration work in Phase 2.)
- Is internet on PC2 needed during demos (AP+STA concurrency question)?
- Demo date → how many rehearsal days to protect?
