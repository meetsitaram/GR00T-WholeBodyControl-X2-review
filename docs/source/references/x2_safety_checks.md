# X2 Safety Checks — reference

All safety mechanisms in the X2 SONIC deploy stack, as designed, tuned,
and **verified on the real robot 2026-08-03/04**. Branch: `safety-checks`.

## 1. Operator e-stop (two-phase, both control surfaces)

**Gesture** (identical on gamepad and VR): hold the **A+X chord** (pad:
cross+square on a PS layout) while **rapidly pumping both triggers**
(L2+R2 / both VR index triggers). A "pump" is one press→release cycle on
*each* trigger, paired within 0.5 s — simultaneous squeeze and
alternating fire both count; hammering a single trigger counts zero.

| Phase | Trip rule | Robot response |
|---|---|---|
| **SOFT** | ≥3 pumps inside 1.0 s, chord held | Abort dance/primitive/locomotion → idle stand. Recoverable. PC3 speaks *"Emergency stop activating."* |
| **DAMP** | last 6 pumps span ≤1.5 s AND ≥1.0 s since first press | Planner latches an `estop` field on every outgoing pose frame → deploy slams stage-2 damping. **Terminal** until deploy restart. PC3 speaks *"Emergency stop. Pure damping engaged."* Robot folds to a kneel (§2). |

Guards: slow pumps (6 over 3–5 s) never damp; chord release before the
6th pump resets and re-arms; chord gaps ≤0.8 s are bridged (WebXR input
sources flap transiently); A+X is **reserved** — a static test fails if
any future binding combines those buttons.

**Wire path (robot)**: pad bridge / VR manager → planner `:5563`
(e-stop outranks source ownership) → planner pose frames (+`estop`
field) → pose watchdog (verbatim forward in LIVE) → deploy
`ZmqPoseInputSource::EstopRequested()` → `ForceTripEstop()` → stage 2.
In the **sim stack** the recorder sits between planner and deploy and
must pass the flag through explicitly (it rebuilds payloads key-by-key
— this was a shipped bug once).

**Regression suite**: `gear_sonic/utils/teleop/test_estop_gesture.py`
(19 tests; every live incident is a named `test_incident_*`) runs as
**stage 0 of `preflight_planner.py`** — a red suite blocks any PC2 ship.
Offline replay of any captured session:
`gear_sonic/scripts/replay_quest3_estop.py <session>/quest3_raw.jsonl`
(record with the stack's `--debug-input`).

## 2. Stage-2 damping profile — and why the paper's value was wrong for X2

The SONIC paper (and the stock G1 deploy) uses a **flat Kp=0, Kd=8
Nm·s/rad on every joint** for the fall-committed damping mode. Shipped
as-is on X2, the damp collapse produced a violent motor whir/growl —
the operator's first live e-stop damp sounded like a malfunction and
ended in a battery pull.

**Root cause**: Kd=8 is tuned for the G1's actuator lineup. On X2 the
value happens to match the *large leg actuators* almost exactly, but
the small, high-gear-ratio joints (wrists, elbows, shoulders, ankle
roll) are over-damped **2–8×** — commanded damping torque far above
what those gearboxes are happy with at back-drive speeds, hence the
whir. A flat constant cannot fit a heterogeneous actuator set.

**Fix — measure the vendor**: the stock AgiBot MC damping mode was
captured on hardware (`x2_damp_capture.py`: 250 Hz joint pos/vel/eff,
247k samples across a commanded slump + hand back-driving of every limb
in damping mode, live-mirrored over wifi so a battery pull cannot erase
evidence). Least-squares `tau = −Kd·vel` per joint; left/right agree to
0.01–0.1 and gravity-adjusted refits match plain fits to 0.01 — these
are firmware constants:

| Joint | Vendor Kd | Ours (flat) | Ratio |
|---|---|---|---|
| Hip pitch / roll / yaw | 9.0 / 8.0 / 8.0 | 8 | ~1× (fine) |
| Knee | 8.0 | 8 | 1× (fine) |
| Ankle pitch / roll | 7.5 / **3.0** | 8 | 1× / **2.7×** |
| Waist yaw / pitch / roll | 5.0 / 6.25 / 4.0 | 8 | 1.3–2× |
| Shoulder pitch / roll / yaw | 4.0 / 5.0 / 3.0 | 8 | 1.6–2.7× |
| Elbow | 3.0 | 8 | 2.7× |
| Wrist yaw / pitch / roll | 3.0 / **1.0** / **1.0** | 8 | **8×** |
| Head | 0.0 (undamped) | 8 | ∞ |

The table ships in `safety.cpp` (`kVendorDampKd`, MJCF joint order) with
one **deliberate departure**: **knees at 4.0** (vendor 8.0) so the legs
are the preferential fold point — the robot drops onto its knees
instead of toppling stiff-legged onto its torso. Hips stay at 9/8 to
slow torso pitch through the fold. Verified live: knees-first fold
(first joint moving, 2.9 rad/s peak), efforts ≤12 Nm, operator verdict
"that was so smooth."

## 3. Tilt watchdog (two-stage)

Stage 1 (recoverable wobble): hold default pose at kd×4 slump.
Stage 2 (fall committed): the §2 damping table, engaged on deep tilt
(>~55° from upright), 0.7 s of stage-1 persistence, or any forced trip
(velocity, e-stop, staleness).

## 4. CONTROL staleness guard

If robot state (leg/waist/arm/head/IMU) goes stale in CONTROL, the
policy is acting on fiction → SAFE_HOLD → stage-2 damping. **Tuning
matters**: first shipped at 150 ms / single-strike (3.3× tighter than
G1's LOW_STATE_ABSENT 500 ms); real aimdk streams can gap past 150 ms
and sim streams never do, so no sim test could catch it. Now **500 ms
AND 3 consecutive ticks**, with a per-stream age report logged on every
strike (`AimdkIo::StateAgeReport`). Lesson: when porting a safety
threshold, port the reference's *number*, not a "hardened" guess —
hardware decides.

## 5. Joint-velocity watchdog

Any measured |dq| > 35 rad/s (`--joint-vel-trip`) force-trips stage 2.
Far above any trained motion (corpus max ~14 rad/s; wrist hardware
limit 20.9) — trips only on genuinely violent motion or sensor garbage.

## 6. Thermal protection (two layers)

* **Deploy C++ monitor**: coil/motor temps ingested per joint; 90 °C
  enter / 85 °C exit hysteresis, logged `THERMAL:` warnings.
* **`x2_thermal_notifier.py` (PC2 daemon, ritual-started)**: audible
  early warning, independent of the deploy (subscribes HAL directly, so
  it works under vendor MC too). Two classes, operator-designed:
  * **CRITICAL** legs+waist, ≥75 °C: PC3 voice *"Warning. Leg motor
    temperature high. Consider resting the robot."* + 5×600 ms pad
    rumble, every 120 s while hot.
  * **UPPER** arms+head, ≥80 °C: softer *"Notice. Arm motor temperature
    elevated."*, no rumble, every 300 s.
  * 3 °C hysteresis; live per-group status every 30 s; readings outside
    (5, 110) °C are sensor-invalid (head publishes a constant 121 °C
    sentinel) — excluded from alerting, logged once.
  * Bring-up protocol: thresholds were first set to 40 °C to verify
    voice + rumble + cadence on hardware, then raised. Reuse this
    pattern for any new alert channel.

## 7. Additional guards

* **NaN action guard**: non-finite policy output → SAFE_HOLD.
* **Deadman**: pad L2+R2 held = sticks live; release = one zero-command.
* **VR input-health monitor**: WebXR source flaps kill button/trigger
  input while pose keeps streaming; manager logs CRITICAL "VR E-STOP
  may be DEAD — use the gamepad" (5 s cadence for 30 s, then 60 s).
* **Pose watchdog** (PC2): upstream silence >100 ms → HOLD → BLEND →
  idle-clip ladder, so the deploy never sees a silent wire.
* **Command watchdog** (planner): no upstream intent → forced IDLE.

## 8. Evidence infrastructure (born from the 2026-08-04 postmortems)

* A battery pull discards **all dirty page cache** — the last ~30–60 s
  of every on-robot log vanished once and faked a "gesture never
  registered" picture. App-level flush ≠ fsync ≠ wire.
* Deploy CSVs flush at 1 Hz; `x2_damp_capture.py` fsyncs at 2 Hz **and
  live-streams every row over ZMQ** — `--mirror tcp://<pc2>:5599` on the
  laptop holds evidence to the last wifi packet.
* All button edges, mode switches, and e-stop lifecycle lines are
  timestamped to the millisecond on both surfaces.
* Postmortem rule: after any power cut, find the common log-cutoff
  horizon **before** treating absence of log lines as absence of events.

## 9. Voice prompts

Generated + staged by `gen_pc3_audio_prompts.py [--stage]` (gTTS →
48 kHz WAV, played via PC3's `aplay -D playback_def` dmix device).
Canonical WAVs live in `gear_sonic_deploy/data/pc3_audio/` — PC3 is not
backed up; the repo is the source of truth.
