# MuJoCo Kitchen / Tabletop Environments for Humanoid Training — Working Notes

A condensed reference compiled from a research conversation (May 2026). Covers
the landscape of kitchen-style environments for humanoid robot training, how
RoboCasa fits in, the LW-BenchHub alternative, what it takes to plug a custom
humanoid into RoboCasa, and minimal MuJoCo XML scenes you can use today as a
lightweight starting point.

---

## 1. The landscape: kitchen environments for humanoids

The "kitchen + robot in MuJoCo" question is muddled by the fact that the most
famous kitchen sim uses a single arm, not a humanoid. A quick disambiguation:

| Project | Engine | Robot(s) | Best for |
|---|---|---|---|
| **Franka Kitchen** (Gymnasium-Robotics) | MuJoCo | 9-DoF Franka Panda arm | Classic IL/RL benchmark; arm only |
| **BiGym** | MuJoCo | Bi-manual mobile humanoid | ~40 home/kitchen tasks, IL demos |
| **HumanoidBench** | MuJoCo | Unitree H1 + 2 Shadow Hands (~101 DoF) | Whole-body RL; not strictly kitchen |
| **RoboCasa** | MuJoCo (via robosuite) | Mobile manipulators, GR-1 humanoid (tabletop) | Generative kitchen scenes, IL/VLA |
| **MuJoCo Playground** | MJX (GPU MuJoCo) | Quadrupeds, humanoids, hands, arms | Clean base for composing your own |
| **LW-BenchHub** | Isaac Lab / PhysX (NOT MuJoCo) | Unitree G1, Franka, ARX, etc. | GPU-parallel RL, photoreal rendering |

**Rules of thumb:**
- Want a humanoid in a kitchen, off the shelf, in MuJoCo → **BiGym**.
- Want a humanoid with dexterous hands for general manipulation in MuJoCo → **HumanoidBench**.
- Okay with arm-only kitchen → **Franka Kitchen**.
- Want massively parallel RL on a humanoid in a kitchen, MuJoCo not required → **LW-BenchHub** (Isaac Lab).
- Want IL/VLA fine-tuning on huge kitchen diversity → **RoboCasa365**.

---

## 2. RoboCasa, in depth

### What it is
A large-scale simulation framework built on **robosuite** (which uses MuJoCo
as its physics engine). Originally released by UT Austin in 2024, now
maintained jointly with NVIDIA. Centered on kitchen environments, with
explicit support for humanoid robots, mobile manipulators, and quadrupeds with
arms.

### Repos to know
1. **`robocasa/robocasa`** — main repo. Latest: **RoboCasa365** (v1.0,
   Feb 2026):
   - 365 tasks (up from 100)
   - 2,500+ kitchen scenes (50 layouts × 50 styles)
   - 3,200+ 3D objects across 150+ categories
   - 600+ hours of human teleop demos + 1,600+ hours of synthetic demos
   - Built-in benchmarking for Diffusion Policy, π (Physical Intelligence),
     and GR00T
   - Cross-embodiment: single-arm mobile, humanoid, quadruped+arm

2. **`robocasa/robocasa-gr1-tabletop-tasks`** — fork specifically for the
   Fourier GR-1 humanoid (the robot NVIDIA targets with GR00T N1/N1.5/N1.6).
   Tasks like:
   - `PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env`
   - `PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env`
   - `PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env`

   Uses the GR-1's arms + waist + Fourier dexterous hands.

### Quick install (humanoid version)

```bash
conda create -c conda-forge -n robocasa python=3.10
conda activate robocasa

# Optional: NVIDIA's GR00T policy stack
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T && pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
cd ..

# robosuite (MuJoCo backend)
git clone https://github.com/ARISE-Initiative/robosuite.git
pip install -e robosuite

# robocasa GR1 tasks
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
pip install -e robocasa-gr1-tabletop-tasks

cd robocasa-gr1-tabletop-tasks
python robocasa/scripts/download_tabletop_assets.py -y
python3 robocasa/scripts/demo_task.py <TASK_NAME>
```

If you use the main `robocasa` repo instead, the API is roughly
`gym.make("robocasa/PickPlaceCounterToCabinet", ...)`.

### Caveats
- Primarily an **imitation-learning / VLA benchmark**, not a from-scratch RL
  benchmark. Most published work uses BC, Diffusion Policy, or VLAs
  (GR00T / π) fine-tuned on demos. PPO/SAC from scratch is hard here:
  high-dim action spaces, long-horizon tasks. Prefer HumanoidBench for pure
  RL.
- The GR-1 tasks are **tabletop only** (fixed base; arms+waist+hands move).
  For a mobile humanoid roaming the kitchen, you'd compose your own task in
  the main repo.
- **MJX/GPU acceleration is not first-class.** Standard MuJoCo through
  robosuite — fine for data generation and rollouts, not ideal for
  massively-parallel GPU RL.
- Only the GR-1 ships as an official humanoid benchmark. Embodiment swapping
  is supported (see §4) but not turnkey.

---

## 3. RoboCasa vs LW-BenchHub

**LW-BenchHub** (Lightwheel AI) is built on **NVIDIA Isaac Lab–Arena**, *not*
MuJoCo. It bundles:
- 268 tasks (Lightwheel-LIBERO-Tasks: 130, Lightwheel-RoboCasa-Tasks: 138)
- 7 robot families / 27 variants — incl. Unitree G1 humanoid, Franka,
  PandaOmron, ARX X7s, Agilex Piper, LeRobot SO-100/101, dual-Panda
- 100 kitchen configurations (10 layouts × 10 styles)
- 21,500-episode demo dataset on HuggingFace
- Teleop pipeline (Vision Pro / Quest / PICO + leader-follower arms)
- RL pipeline integrating with `rsl-rl` and `skrl`
- Decoupled server-client policy API

The "RoboCasa" inside LW-BenchHub is a **port** of RoboCasa tasks into Isaac
Lab–Arena — not the original MuJoCo-based RoboCasa.

### Where LW-BenchHub wins
- **Massively parallel GPU RL.** Isaac Lab → GPU-accelerated PhysX, thousands
  of parallel envs. RoboCasa is CPU MuJoCo via robosuite.
- **Photorealistic rendering** via Isaac Sim's RTX renderer — better for
  vision sim-to-real.
- **Whole-body humanoid control out of the box.** Unitree G1 ships with a
  decoupled WBC, locomotion-aware base commands (`lin_x_local`, `ang_z_local`,
  base height, torso roll/pitch/yaw), and synchronized dual-arm + dexterous
  hand control.
- **Modern teleop hardware** — native Vision Pro / Quest / PICO + leader-
  follower arms.

### Where RoboCasa still wins
- **You wanted MuJoCo.** LW-BenchHub is Isaac Lab/PhysX.
- **Maturity & community.** RoboCasa is from UT Austin + NVIDIA, RSS 2024 +
  ICLR 2026 follow-up, broad VLA adoption (GR00T N1.5/N1.6, π, Diffusion
  Policy). LW-BenchHub is newer (v1.0.0 Dec 2025, ~107 stars at writing).
- **Scale of scenes & assets.** RoboCasa365 has 2,500 scenes, 3,200+ objects
  vs. 100 configurations.
- **Hardware floor.** Isaac Lab realistically wants RTX GPU + CUDA 12.8,
  driver 570.x. RoboCasa runs on a laptop.
- **Demo data scale.** RoboCasa365: 600h human + 1,600h synthetic vs. 21,500
  episodes (more diverse across robots though).

### TL;DR pick
- **RL on a humanoid in a kitchen** → LW-BenchHub.
- **IL / VLA fine-tuning, max scene diversity, mature ecosystem** → RoboCasa365.
- **Strict MuJoCo + humanoid + kitchen** → neither is ideal; HumanoidBench +
  custom kitchen assets, or BiGym.

---

## 4. Loading a custom humanoid into RoboCasa

**Yes, supported — but documented turnkey it is not.** Custom robot loading
is inherited from `robosuite v1.5+` (Oct 2024 headline feature: "Custom Robot
Composition"), pulled into RoboCasa v0.2 the same week.

### How it actually works
Robots in robosuite decompose into:
- `RobotModel` + `RobotBaseModel` + `GripperModel` + `Controller(s)`

Each is an MJCF (XML) asset + Python class. To add a custom humanoid:

1. **Convert your robot to MuJoCo MJCF.** URDF alone isn't enough. Use
   `mujoco.compiler` (URDF→MJCF) or hand-author using **MuJoCo Menagerie** as
   reference. Menagerie ships high-quality MJCF for many humanoids
   (Unitree H1/G1, Apptronik Apollo, Booster T1, ...) which often spares you
   the conversion entirely.

2. **Write a `RobotModel` Python class.** Subclass:
   - `LeggedRobot` for bipedal humanoids (most common)
   - `WheeledRobot`
   - `FixedBaseRobot`

   Declare default arm joints, gripper attachment sites, sensor sites, eef
   name, action ranges, etc.

3. **Register the class** with `robosuite.robots.register_robot_class(...)`
   so it gets a string name usable as `robot_type="YourHumanoid"`.

4. **Define a composite controller config.** For a humanoid: whole-body
   controller + per-part controllers (left arm, right arm, torso/waist, base).
   v1.5's composite controllers are designed for this — each part gets its
   own controller (joint/IK/OSC) coordinated by the WBC.

5. **Add grippers / hands.** Subclass `GripperModel` per hand. Inspire Hands
   and the GR-1's Fourier hands are existing examples.

6. **Plug into a RoboCasa task:**
   ```python
   env = suite.make(
       "PickPlaceCounterToCabinet",
       robots=["YourHumanoid"],
       ...
   )
   ```
   Most kitchen tasks are robot-agnostic; you'll likely tune spawn position
   and reach-related task constraints.

### Realistic warnings
- **Docs are sparse.** Official "create your own robot" page largely says
  "see `robosuite_models` repo" + a 2024 Google Doc draft. Best learning
  resource: read existing implementations in
  `robosuite/models/robots/` and the `robosuite-models` external repo.
- **GR-1 is your best reference.** Only humanoid shipped natively, 24 DoF,
  variants `GR1FixedLowerBody`, `GR1FloatingBody`, `GR1ArmsOnly`. Cloning
  the GR-1 class and swapping the MJCF/joint names is the fastest path.
- **RoboCasa-specific quirks.** Beyond robosuite, RoboCasa has assumptions
  about robot placement in kitchens (spawn poses, base height, reach radius
  for sampling object placements). May need to override
  `_get_robot_base_pose()`-style hooks or pass `robot_offset`. Look at how
  `GR1FixedLowerBody` is wired in `robocasa-gr1-tabletop-tasks`.
- **Known pain point.** An open issue on NVIDIA's `Isaac-GR00T` repo asks
  exactly how to swap Franka in for GR-1 — as of mid-2025 there wasn't a
  clean published recipe.

### Recommended workflow
1. Get your humanoid into Menagerie-style MJCF first; verify in
   `mujoco.viewer`.
2. Clone `robosuite_models`; study GR-1 end-to-end.
3. Copy the GR-1 directory; swap XML, rename class, update joint names/DoF.
4. Test in plain robosuite with a trivial task (e.g. `Lift`) before going
   to RoboCasa.
5. Once working in robosuite:
   `python -m robocasa.demos.demo_teleop --robots YourHumanoid`
6. Override placement methods in the RoboCasa task if reach/spawn issues
   appear.

---

## 5. Going lightweight: hand-rolled MuJoCo scenes

If full RoboCasa is overkill and you just want "stuff around the robot,"
plain MJCF is the fastest path. This is genuinely 30 lines of XML.

### 5.1 Minimal table + cube

```xml
<mujoco model="table_with_cube">
  <option gravity="0 0 -9.81"/>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>

    <body name="table" pos="0 0 0.4">
      <geom name="tabletop" type="box" size="0.5 0.3 0.02"
            rgba="0.6 0.4 0.2 1"/>
      <geom name="leg1" type="box" size="0.02 0.02 0.2"
            pos=" 0.45  0.25 -0.22" rgba="0.4 0.25 0.1 1"/>
      <geom name="leg2" type="box" size="0.02 0.02 0.2"
            pos="-0.45  0.25 -0.22" rgba="0.4 0.25 0.1 1"/>
      <geom name="leg3" type="box" size="0.02 0.02 0.2"
            pos=" 0.45 -0.25 -0.22" rgba="0.4 0.25 0.1 1"/>
      <geom name="leg4" type="box" size="0.02 0.02 0.2"
            pos="-0.45 -0.25 -0.22" rgba="0.4 0.25 0.1 1"/>
    </body>

    <body name="cube" pos="0 0 0.45">
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.03 0.03 0.03"
            rgba="1 0 0 1" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
```

Run with:
```bash
python -m mujoco.viewer --mjcf=scene.xml
```

**Gotchas:**
- `size` is **half-extents** for boxes (so `0.5 0.3 0.02` = 1.0×0.6×0.04 m).
- `<freejoint/>` is what makes the cube physical. Without it the cube is
  welded to the world.
- Z-positions must add up. Spawn objects at the *exact* surface height or
  slightly above; spawn below and the simulator will explode on wake-up.

### 5.2 Richer "world" file (table + walls + objects + cameras)

```xml
<mujoco model="humanoid_world">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
    <material name="wood"  rgba="0.55 0.35 0.15 1"/>
    <material name="metal" rgba="0.7 0.7 0.75 1" specular="0.6"/>
    <material name="red"   rgba="0.85 0.15 0.15 1"/>
    <material name="green" rgba="0.15 0.7 0.25 1"/>
    <material name="blue"  rgba="0.15 0.3 0.85 1"/>
  </asset>

  <default>
    <default class="obstacle">
      <geom type="box" material="wood" friction="1 0.05 0.001"/>
    </default>
    <default class="manipulable">
      <geom friction="1.5 0.1 0.001" condim="4" priority="1"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <light pos="2 2 3" dir="-1 -1 -1" diffuse="0.3 0.3 0.3"/>

    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"
          condim="3" friction="1 0.005 0.0001"/>

    <geom name="wall_back"  type="plane" size="3 2 0.1" pos=" 0 -2 1"
          zaxis="0 1 0" rgba="0.85 0.85 0.85 1"/>
    <geom name="wall_left"  type="plane" size="2 2 0.1" pos="-3 0 1"
          zaxis="1 0 0" rgba="0.9 0.9 0.9 1"/>
    <geom name="wall_right" type="plane" size="2 2 0.1" pos=" 3 0 1"
          zaxis="-1 0 0" rgba="0.9 0.9 0.9 1"/>

    <camera name="front" pos="0 -2.0 1.4" xyaxes="1 0 0 0 0.5 0.87"/>
    <camera name="side"  pos="2.5 0 1.4"  xyaxes="0 -1 0 0.5 0 0.87"/>
    <camera name="top"   pos="0 0 3.5"    xyaxes="1 0 0 0 1 0"/>

    <!-- Table: tabletop top surface at z = 0.775 -->
    <body name="table" pos="0.7 0 0.0">
      <geom name="tabletop" class="obstacle" size="0.45 0.35 0.025"
            pos="0 0 0.75" material="wood"/>
      <geom name="leg_fl" class="obstacle" size="0.025 0.025 0.365"
            pos=" 0.40  0.30 0.365" material="metal"/>
      <geom name="leg_fr" class="obstacle" size="0.025 0.025 0.365"
            pos=" 0.40 -0.30 0.365" material="metal"/>
      <geom name="leg_bl" class="obstacle" size="0.025 0.025 0.365"
            pos="-0.40  0.30 0.365" material="metal"/>
      <geom name="leg_br" class="obstacle" size="0.025 0.025 0.365"
            pos="-0.40 -0.30 0.365" material="metal"/>
    </body>

    <body name="cube" pos="0.7 -0.15 0.80">
      <freejoint/>
      <geom name="cube_geom" class="manipulable" type="box"
            size="0.025 0.025 0.025" material="red" mass="0.15"/>
    </body>

    <body name="cylinder" pos="0.7 0.0 0.81">
      <freejoint/>
      <geom name="cyl_geom" class="manipulable" type="cylinder"
            size="0.035 0.035" material="green" mass="0.2"/>
    </body>

    <body name="ball" pos="0.7 0.18 0.81">
      <freejoint/>
      <geom name="ball_geom" class="manipulable" type="sphere"
            size="0.03" material="blue" mass="0.1"/>
    </body>

    <site name="goal" pos="0.55 0.25 0.776" size="0.04 0.001"
          type="cylinder" rgba="1 1 0 0.5"/>

    <!-- Drop your humanoid here:
         <include file="my_humanoid.xml"/>  -->
  </worldbody>
</mujoco>
```

### 5.3 Minimal "place a cube into a bowl" scene

MuJoCo primitives don't include a hollow vessel. Three options for a bowl:
1. **Composite from primitives** (below) — cheap, parametric, no external
   files.
2. **Use a mesh** — author in Blender, export STL, reference via
   `<mesh file="bowl.stl"/>`. Need convex decomposition for collisions.
3. **Use a single box "tray"** — simplest of all.

Composite bowl approach:

```xml
<mujoco model="cube_and_bowl">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
    <material name="wood"  rgba="0.55 0.35 0.15 1"/>
    <material name="red"   rgba="0.85 0.15 0.15 1"/>
    <material name="white" rgba="0.92 0.92 0.92 1" specular="0.3"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>

    <body name="table" pos="0.7 0 0">
      <geom name="tabletop" type="box" size="0.45 0.35 0.025"
            pos="0 0 0.75" material="wood"/>
      <geom name="leg_fl" type="box" size="0.025 0.025 0.365"
            pos=" 0.40  0.30 0.365" material="wood"/>
      <geom name="leg_fr" type="box" size="0.025 0.025 0.365"
            pos=" 0.40 -0.30 0.365" material="wood"/>
      <geom name="leg_bl" type="box" size="0.025 0.025 0.365"
            pos="-0.40  0.30 0.365" material="wood"/>
      <geom name="leg_br" type="box" size="0.025 0.025 0.365"
            pos="-0.40 -0.30 0.365" material="wood"/>
    </body>

    <body name="cube" pos="0.7 -0.12 0.80">
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025"
            material="red" mass="0.1"
            friction="1.5 0.1 0.001" condim="4" priority="1"/>
    </body>

    <!-- Bowl: base disc + 12-segment ring for the wall.
         Inner radius ~7cm, wall height 4cm. -->
    <body name="bowl" pos="0.7 0.12 0.785">
      <freejoint/>
      <geom name="bowl_base" type="cylinder" size="0.085 0.005"
            pos="0 0 0" material="white" mass="0.15"
            friction="1 0.05 0.001" condim="4"/>
      <geom name="w0"  type="box" size="0.022 0.005 0.020"
            pos=" 0.075  0.000 0.025" euler="0 0 0"
            material="white" mass="0.01" condim="4"/>
      <geom name="w1"  type="box" size="0.022 0.005 0.020"
            pos=" 0.065  0.0375 0.025" euler="0 0 0.5236"
            material="white" mass="0.01" condim="4"/>
      <geom name="w2"  type="box" size="0.022 0.005 0.020"
            pos=" 0.0375 0.065 0.025" euler="0 0 1.0472"
            material="white" mass="0.01" condim="4"/>
      <geom name="w3"  type="box" size="0.022 0.005 0.020"
            pos=" 0.000  0.075 0.025" euler="0 0 1.5708"
            material="white" mass="0.01" condim="4"/>
      <geom name="w4"  type="box" size="0.022 0.005 0.020"
            pos="-0.0375 0.065 0.025" euler="0 0 2.0944"
            material="white" mass="0.01" condim="4"/>
      <geom name="w5"  type="box" size="0.022 0.005 0.020"
            pos="-0.065  0.0375 0.025" euler="0 0 2.6180"
            material="white" mass="0.01" condim="4"/>
      <geom name="w6"  type="box" size="0.022 0.005 0.020"
            pos="-0.075  0.000 0.025" euler="0 0 3.1416"
            material="white" mass="0.01" condim="4"/>
      <geom name="w7"  type="box" size="0.022 0.005 0.020"
            pos="-0.065 -0.0375 0.025" euler="0 0 3.6652"
            material="white" mass="0.01" condim="4"/>
      <geom name="w8"  type="box" size="0.022 0.005 0.020"
            pos="-0.0375 -0.065 0.025" euler="0 0 4.1888"
            material="white" mass="0.01" condim="4"/>
      <geom name="w9"  type="box" size="0.022 0.005 0.020"
            pos=" 0.000 -0.075 0.025" euler="0 0 4.7124"
            material="white" mass="0.01" condim="4"/>
      <geom name="w10" type="box" size="0.022 0.005 0.020"
            pos=" 0.0375 -0.065 0.025" euler="0 0 5.2360"
            material="white" mass="0.01" condim="4"/>
      <geom name="w11" type="box" size="0.022 0.005 0.020"
            pos=" 0.065 -0.0375 0.025" euler="0 0 5.7596"
            material="white" mass="0.01" condim="4"/>
    </body>

    <!-- Drop your humanoid here:
         <include file="my_humanoid.xml"/>  -->
  </worldbody>
</mujoco>
```

**Notes:**
- Bowl is a single rigid body (all 13 geoms move together).
- Drop the `<freejoint/>` from the bowl body to weld it in place.
- Bowl ~0.16 kg total, cube 0.1 kg — tune masses per your humanoid's grip.
- `condim="4"` + `priority="1"` on manipulables gives better friction with
  fingertips (less "skittering").
- `integrator="implicitfast"` is the modern recommendation for
  contact-heavy manipulation scenes.

### 5.4 Two ways to insert your humanoid

**Option A — `<include>` (cleanest):**
```xml
<include file="my_humanoid.xml"/>
```
Place inside `<worldbody>`, OR at the top level — MuJoCo will merge sections
appropriately. Cleanest pattern: assets/defaults at top level, the actual
`<body>` in a separate include that goes inside `<worldbody>`.

**Option B — paste inline.** Copy the humanoid `<body>` directly where you
want it; merge the humanoid file's `<asset>`, `<default>`, `<actuator>`
blocks into your scene's top level.

### 5.5 Spawn / sizing tuning checklist
- Humanoid root pose: ~`(0, 0, <feet height>)`, facing +x by default.
- Tabletop top surface in §5.3 is at **z = 0.775** (table at z=0, tabletop
  center z=0.75, half-thickness 0.025). Raise/lower the table body's `pos`
  Z to fit your humanoid's reach.
- With humanoid at origin and table at x=0.7 → ~70 cm reach. Move the table
  closer (x=0.5) for shorter-armed robots.

---

## 6. "Simple but richer" example projects worth studying

Curated by amount of complexity over a bare scene:

### Closest to a bare scene
- **MuJoCo Menagerie `scene.xml`** files — every robot package ships a small
  scene with textured ground, skybox, atmospheric haze. Worth opening for
  asset/lighting structure:
  - `mujoco_menagerie/franka_emika_panda/scene.xml`
  - `mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml` ← **best
    starting reference** (Panda + table + cube, MJX-ready, ~100 lines)
  - `mujoco_menagerie/robotiq_2f85/scene.xml` (gripper + hanging box)

### One step up — MJX manipulation playground
**MuJoCo Playground** `manipulation/franka_emika_panda/`:
- `pick_cube` — Panda + table + cube
- `pick_cube_orientation` — same with orientation goal
- `open_cabinet` — articulated drawer

~150 lines of Python + XML each. Great template if you want a Gym-style env
wrapped around a clean MuJoCo scene.

### Tutorial repos
- **`jeongeun980906/lerobot-mujoco-tutorial`** — pick-mug-place-on-plate
  with the OMY arm. Includes WASD/arrow-key teleop, then trains with
  LeRobot. Builds the MJCF at init time by `<include>`-ing per-object
  fragments from `./objects/<name>/model_new.xml`.
- **`tlpss/mujoco-sim`** — uses dm_control `composer` Entity abstraction
  (planar pushing, button-press, pointmass reach). Cleaner separation:
  `entities/`, `arenas/`, `robots/`. Good for programmatic scene generation.

### Procedural scene generation
- **`dm_control.mjcf` (PyMJCF)** — DeepMind's procedural API. Build the
  MJCF tree as Python objects, attach grippers, instantiate N copies, and
  compile. Great for randomized scenes per episode.
- **MuJoCo 3.2+'s native `mjSpec` API** — same idea, now built into MuJoCo
  proper. See GitHub discussion #2063 for attaching a hand to an arm.

### Robosuite's canonical "table + cube"
`robosuite.environments.manipulation.lift.Lift`. Reading its source
side-by-side with raw MJCF shows what robosuite's abstractions actually buy
you: random placement, arena re-use, observation extraction.

### Recommended learning path
1. Open `mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml`.
2. Look at MuJoCo Playground's `pick_cube` env to see the same scene wrapped
   as a JAX/MJX RL environment.
3. Drop in your own humanoid the same way.

---

## 7. One-line decision matrix

| If your goal is... | Use |
|---|---|
| Off-the-shelf humanoid in a kitchen, MuJoCo | **BiGym** |
| Humanoid + dexterous hands, general manip, MuJoCo | **HumanoidBench** |
| Arm-only kitchen, MuJoCo | **Franka Kitchen** |
| GPU-parallel humanoid RL in a kitchen (any sim) | **LW-BenchHub** (Isaac Lab) |
| IL/VLA fine-tuning, max kitchen diversity | **RoboCasa365** |
| Custom humanoid in RoboCasa | Subclass `LeggedRobot`, copy GR-1 pattern |
| Compose your own scene around a custom humanoid | Hand-rolled MJCF (§5) → MuJoCo Playground style env |

---

## 8. Reference links

- RoboCasa: https://github.com/robocasa/robocasa
- RoboCasa GR-1 tabletop tasks: https://github.com/robocasa/robocasa-gr1-tabletop-tasks
- robosuite: https://github.com/ARISE-Initiative/robosuite
- robosuite-models (for custom robot examples): https://github.com/ARISE-Initiative/robosuite-models
- NVIDIA Isaac-GR00T: https://github.com/NVIDIA/Isaac-GR00T
- LW-BenchHub: https://github.com/LightwheelAI/LW-BenchHub
- BiGym: https://github.com/chernyadev/bigym
- HumanoidBench: https://github.com/carlosferrazza/humanoid-bench
- Gymnasium-Robotics (Franka Kitchen): https://github.com/Farama-Foundation/Gymnasium-Robotics
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
- dm_control (PyMJCF): https://github.com/google-deepmind/dm_control
- LeRobot MuJoCo tutorial: https://github.com/jeongeun980906/lerobot-mujoco-tutorial
- mujoco-sim (dm_control composer): https://github.com/tlpss/mujoco-sim
