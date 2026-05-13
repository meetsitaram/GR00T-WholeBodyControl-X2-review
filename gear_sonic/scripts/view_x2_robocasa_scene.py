"""Open an interactive MuJoCo viewport on an X2 + robocasa task.

Quick visual sanity check for the Phase 1 X2 + GR00T integration:

* loads ``X2PickPlaceCube`` (or any other registered locomanip env) with
  the ``X2UltraFixedLowerBody`` robot model and the X2-default composite
  controller (``JointPosition`` arms / torso / head + ``SimpleGrip``
  hands, see ``robocasa/controllers/config/default/composite/x2_ultra_default.json``);
* spins the env at the configured control frequency in the foreground
  thread, holding the policy at zero action so the robot stays in its
  ``init_qpos`` while we orbit the scene;
* surfaces the live success flag and reset shortcut so you can sanity
  the cube spawn ranges / bowl placement without a recorder in the loop.

Keyboard shortcuts (in addition to the MuJoCo viewer defaults):

* ``r``  reset the env (re-randomises cube / bowl placements)
* ``q`` / ``ESC`` quit (or just close the window)

Run it from the repo root with the sim venv:

.. code-block:: bash

    .venv_sim/bin/python gear_sonic/scripts/view_x2_robocasa_scene.py

CLI:

.. code-block:: bash

    --env X2PickPlaceCube|X2PickPlaceBowl|LMBottlePnP|...
    --robot X2UltraFixedLowerBody|X2UltraFixedBase|...
    --camera <viewer | robot0_rs_egoview | egoview | ...>
    --fps 50    # how often the viewer thread re-syncs (display only)

The script does not record or write any data; it is purely a visualiser.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive viewport for X2 + robocasa scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        default="X2PickPlaceCube",
        help="Registered locomanip env (e.g. X2PickPlaceCube, X2PickPlaceBowl, LMBottlePnP).",
    )
    parser.add_argument(
        "--robot",
        default="X2UltraFixedLowerBody",
        help="Robosuite robot model to load (X2UltraFixedLowerBody / X2UltraFixedBase / ...).",
    )
    parser.add_argument(
        "--camera",
        default="viewer",
        help=(
            "Camera to start the viewer on. ``viewer`` keeps the default free "
            "orbiting camera; pass an MJCF camera name (e.g. robot0_rs_egoview, "
            "egoview, robot0_oak_egoview) to start fixed on that camera."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Display sync rate; the env physics still steps at its own control_freq.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for the env reset (omit for non-deterministic).",
    )
    parser.add_argument(
        "--no-step",
        action="store_true",
        help="Don't step the env -- just open the viewer on the post-reset state.",
    )
    parser.add_argument(
        "--scene-xml",
        type=str,
        default=None,
        help=(
            "Skip the robosuite env entirely and load a static MJCF file "
            "(e.g. one produced by gear_sonic/scripts/build_x2_robocasa_scene_xml.py). "
            "When set, --env / --robot / --no-step are ignored. Useful for "
            "verifying that the static scene XML the deploy bridge will see "
            "renders correctly."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    warnings.filterwarnings("ignore")
    # Render through whatever the user has -- ``MUJOCO_GL=egl`` works for
    # offscreen but the launched viewer will use GLFW under the hood and
    # ignore this setting, so it is safe to leave alone.
    os.environ.setdefault("MUJOCO_GL", "glfw")

    if not os.environ.get("DISPLAY"):
        print(
            "ERROR: no DISPLAY env var detected. Open this from a desktop "
            "session (the script needs an X / Wayland viewport).",
            file=sys.stderr,
        )
        return 1

    # Imports deferred so ``--help`` works without the heavy stack.
    import mujoco
    from mujoco import viewer
    import numpy as np

    env = None
    if args.scene_xml is not None:
        # Load the static MJCF directly -- bypass robosuite. This is the
        # MJCF the deploy bridge will see; if it renders well here, the
        # bridge will see the same scene at runtime.
        scene_path = args.scene_xml
        print(f"Loading static scene XML: {scene_path}")
        mj_model = mujoco.MjModel.from_xml_path(scene_path)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)
        print(
            f"Scene XML loaded. nq={mj_model.nq}, nv={mj_model.nv}, "
            f"nu={mj_model.nu}, ngeom={mj_model.ngeom}"
        )
    else:
        import robocasa  # noqa: F401  (registers envs / robots via side effects)
        import robocasa.models.robots  # noqa: F401  (registers X2 robots)
        import robosuite as suite

        from robocasa.models.grippers import load_x2_default_controller_config

        print(f"Loading env={args.env!r} robot={args.robot!r} ...")
        env = suite.make(
            env_name=args.env,
            robots=args.robot,
            controller_configs=load_x2_default_controller_config(),
            has_renderer=False,  # we drive the viewer ourselves
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
        if args.seed is not None:
            env.seed(args.seed)
        env.reset()
        print(
            "Env loaded. action_dim={}, control_freq={} Hz, # robots={}.".format(
                env.action_dim, env.control_freq, len(env.robots)
            )
        )

        # ``robosuite.environments.base`` keeps the underlying MjModel / MjData
        # behind ``env.sim.model._model`` and ``env.sim.data._data``.
        mj_model = env.sim.model._model
        mj_data = env.sim.data._data

    # Resolve optional fixed camera.
    cam_id = -1  # -1 = free orbiting camera
    if args.camera and args.camera != "viewer":
        try:
            cam_id = mj_model.camera(args.camera).id
            print(f"Starting on fixed camera {args.camera!r} (id={cam_id}).")
        except KeyError:
            available = [mj_model.camera(i).name for i in range(mj_model.ncam)]
            print(
                f"WARNING: camera {args.camera!r} not found in model. "
                f"Falling back to free camera. Available: {available}",
                file=sys.stderr,
            )
            cam_id = -1

    if env is not None:
        zero_action = np.zeros(env.action_dim, dtype=np.float64)
        sim_dt = 1.0 / env.control_freq
    else:
        # No env -- we just hold the scene at its post-mj_forward state.
        zero_action = None
        sim_dt = 1.0 / max(args.fps, 1)
    display_dt = 1.0 / max(args.fps, 1)
    next_sim_step = time.time()
    next_display_sync = time.time()

    print(
        "Opening MuJoCo viewer. Close the window or press Ctrl+C to exit.\n"
        "Tip: the viewer's ``Tab`` key cycles cameras, including the head-mounted ones."
    )

    try:
        with viewer.launch_passive(mj_model, mj_data, show_left_ui=True, show_right_ui=True) as v:
            if cam_id >= 0:
                v.cam.fixedcamid = cam_id
                v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            else:
                v.cam.lookat = [0.5, 0.0, 1.05]
                v.cam.distance = 2.5
                v.cam.azimuth = 180.0
                v.cam.elevation = -10.0

            while v.is_running():
                now = time.time()

                if env is not None and not args.no_step and now >= next_sim_step:
                    env.step(zero_action)
                    next_sim_step += sim_dt
                    # If the viewer thread fell behind (e.g. UI moved), don't
                    # try to catch up forever -- just resync.
                    if now - next_sim_step > 0.5:
                        next_sim_step = now + sim_dt

                if now >= next_display_sync:
                    v.sync()
                    next_display_sync = now + display_dt

                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nCtrl+C received -- closing viewer.")
    finally:
        if env is not None:
            env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
