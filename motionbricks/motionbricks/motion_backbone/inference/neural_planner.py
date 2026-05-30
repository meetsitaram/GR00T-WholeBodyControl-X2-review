"""Robot-agnostic streaming wrapper around motion_inference for velocity-intent control.

This module is the neural counterpart to the G1 demo's `full_navigation_agent`
([`motionbricks/motionbricks/motion_backbone/demo/full_agent.py`](../../demo/full_agent.py)),
stripped of G1-specific clip blendspace / WASD controller / spring-root world-target
model. What remains is the robot-agnostic predict-and-decode core:

  - Maintain a ring buffer of upcoming MuJoCo qpos frames decoded from the network.
  - On replan, take the last 4 qpos frames as past context, build the predict()
    input tensors (8-frame constraint window: past 4 + target 4), and feed a
    desired velocity intent (yaw_rate, vel_x, vel_z, hip_height) as the
    target_local_root_values constraint.
  - Mask off target_local_poses so the VQVAE free-samples poses consistent with
    the velocity constraint (i.e., no target keyframe pose to interpolate to).
  - Decode pred_global_poses -> MuJoCo qpos via the robot-specific converter
    (e.g. X2MujocoQposConverter, G1's get_mujoco_converter); the converter is
    the only robot-specific dependency.

The 8-frame constraint window and the constraint-mask convention follow the
inference contract documented in
[`motion_inference.py`](motion_inference.py):

  - First 4 frames (numConstrainFrames=NUM_FRAMES_PER_TOKEN=4): past/current
    context; all has_* masks must be True for these.
  - Last 4 frames: target keyframe at the end of the predicted motion.
  - num_tokens drives how many tokens (=4 frames each) the model generates
    between the past context and the target keyframe.

For Quest3 velocity-intent teleop we provide:
  - global_root_values past 4: from `_canonicalize_mujoco_qpos`'d context
  - local_root_values past 4: differenced from context positions (matches the
    layout in `LocalRootLocalBody.compute_root_rep_from_root_pos_and_rot`)
  - local_root_values target 4: filled from the velocity intent (broadcast)
  - local_poses past 4: from converted context joint positions+rotations
  - local_poses target 4: zeros, but has_local_poses[:, -4:] = False (masked)
  - has_local_root_values[:, NUM_FRAMES_PER_TOKEN-1] = False (last differenced
    velocity in the context window is invalid; see full_agent.py:457)
  - has_global_root_values[:, -4:] = False (no target world-frame position)

Constructor takes the robot-specific MuJoCo converter (must implement
`convert_motion_features_to_mujoco_qpos` and
`convert_mujoco_qpos_to_motion_transforms`) plus the `motion_inference` wrapper.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch as t
from torch import nn

from motionbricks.geometry.quaternions import matrix_to_quaternion
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from motionbricks.motionlib.core.utils.rotations import (
    matrix_to_cont6d,
    quaternion_to_matrix,
)


# ---------------------------------------------------------------------------
# Z-axis rotation helper (kept verbatim from full_agent.py for MuJoCo qpos
# canonicalization, where the gravity axis in the wire is +z).
# ---------------------------------------------------------------------------


def angle_to_Z_rotation_matrix(angle: t.Tensor) -> t.Tensor:
    """Rotation matrix around Z-axis for Z-up MuJoCo qpos canonicalization."""
    cos, sin = t.cos(angle), t.sin(angle)
    one, zero = t.ones_like(angle), t.zeros_like(angle)
    mat = t.stack((cos, -sin, zero, sin, cos, zero, zero, zero, one), -1)
    mat = mat.reshape(angle.shape + (3, 3))
    return mat


class NeuralPlannerCore(nn.Module):
    """Streaming neural locomotion planner driven by velocity intent.

    Lifecycle:

        core = NeuralPlannerCore(inferencer, converter)
        core.reset(init_mujoco_qpos)              # seed ring buffer with a stand frame
        while True:
            qpos = core.get_next_frame()           # pop one [38]-D qpos per tick
            if core.should_replan():
                core.replan_with_velocity(intent_vector_4d)
    """

    NUM_FRAMES_PER_TOKEN: int = 4
    DEFAULT_PRED_OFFSETS: int = 4
    NUM_MIN_FRAMES_IN_BUFFER: int = 64
    DEFAULT_PLANNING_HORIZON_S: float = 1.0

    def __init__(
        self,
        inferencer: motion_inference,
        converter: nn.Module,
        device: str = "cuda",
        pred_offsets: int = DEFAULT_PRED_OFFSETS,
        filter_qpos: bool = True,
        force_canonicalization: bool = True,
        skip_ending_target_cond: bool = True,
        replan_threshold_frames: int = 16,
    ) -> None:
        super().__init__()
        self._inferencer = inferencer.eval().to(device)
        self._motion_rep = deepcopy(inferencer.motion_rep).to(device)
        self._converter = converter.to(device)
        self._device = device
        self._fps = self._motion_rep.fps

        self.PRED_OFFSETS = pred_offsets
        self.FILTER_QPOS = filter_qpos
        self.FORCE_CANONICALIZATION = force_canonicalization
        self.SKIP_ENDING_TARGET_COND = skip_ending_target_cond
        self.REPLAN_THRESHOLD_FRAMES = replan_threshold_frames

        self.frames: dict = {
            "model_features": None,  # [B, T, feat_dim]
            "mujoco_qpos": None,     # [B, T, qpos_dim]
        }
        self._current_frame_idx = 0
        self._qpos_dim: Optional[int] = None

        # Diagnostic hook. Optional callable invoked inside _predict after
        # the default has_* masks are constructed but before predict() is
        # called. Signature:
        #     fn(has_global_root_values, has_local_root_values,
        #        has_local_poses, NUM_FT) -> (has_g, has_l, has_p)
        # Use this from diagnostic scripts to simulate training-time
        # keyframe-density distributions and isolate train/infer
        # distribution mismatches. Do NOT set this in production code.
        self.diagnostic_mask_hook = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def fps(self) -> float:
        return float(self._fps)

    @property
    def current_frame_idx(self) -> int:
        return self._current_frame_idx

    @property
    def frames_remaining(self) -> int:
        """How many frames are left in the buffer after the current pop index."""
        if self.frames["mujoco_qpos"] is None:
            return 0
        return int(self.frames["mujoco_qpos"].shape[1]) - self._current_frame_idx

    def reset(self, init_mujoco_qpos: t.Tensor) -> None:
        """Seed the ring buffer with NUM_MIN_FRAMES_IN_BUFFER copies of a stand pose.

        Args:
            init_mujoco_qpos: [qpos_dim] float tensor (e.g. [38] for X2).
                Will be tiled to fill the ring buffer.
        """
        if init_mujoco_qpos.ndim == 1:
            init_mujoco_qpos = init_mujoco_qpos[None, None, :]  # [1, 1, D]
        elif init_mujoco_qpos.ndim == 2:
            init_mujoco_qpos = init_mujoco_qpos[None, :, :]
        init_mujoco_qpos = init_mujoco_qpos.to(self._device).float()
        self._qpos_dim = int(init_mujoco_qpos.shape[-1])

        tile_n = self.NUM_MIN_FRAMES_IN_BUFFER
        if init_mujoco_qpos.shape[1] < tile_n:
            init_mujoco_qpos = t.cat(
                [init_mujoco_qpos]
                + [init_mujoco_qpos[:, -1:, :]] * (tile_n - init_mujoco_qpos.shape[1]),
                dim=1,
            )

        self.frames["mujoco_qpos"] = init_mujoco_qpos
        self.frames["model_features"] = None  # not used until first replan
        self._current_frame_idx = 0

    def should_replan(self) -> bool:
        """True when the ring buffer's tail is closer than REPLAN_THRESHOLD_FRAMES."""
        return self.frames_remaining <= self.REPLAN_THRESHOLD_FRAMES

    def get_next_frame(self) -> t.Tensor:
        """Pop the next [qpos_dim] frame and advance the read cursor.

        Clamps at the buffer tail so we never index past the end; the daemon
        should call replan_with_velocity() before this returns a duplicate.
        """
        if self.frames["mujoco_qpos"] is None:
            raise RuntimeError("NeuralPlannerCore.get_next_frame() called before reset()")
        buf = self.frames["mujoco_qpos"]
        idx = self._current_frame_idx
        # Clamp so the buffer's final frame repeats if we ran past replan threshold.
        idx_clamped = max(0, min(idx, buf.shape[1] - 1))
        self._current_frame_idx = min(idx + 1, buf.shape[1] - 1)
        return buf[0, idx_clamped].detach()

    def get_context_mujoco_qpos(self) -> t.Tensor:
        """Return last NUM_FRAMES_PER_TOKEN frames before the current cursor.

        Matches `full_navigation_agent.get_context_mujoco_qpos` exactly so the
        rest of the predict-and-decode pipeline transplants verbatim.
        """
        idx = self._current_frame_idx
        buf = self.frames["mujoco_qpos"]
        indices = [
            max(0, min(idx - self.NUM_FRAMES_PER_TOKEN + i + self.PRED_OFFSETS, buf.shape[1] - 1))
            for i in range(self.NUM_FRAMES_PER_TOKEN)
        ]
        return buf[:, indices, :].to(self._device)

    # ------------------------------------------------------------------
    # Velocity-intent replan
    # ------------------------------------------------------------------

    @t.no_grad()
    def replan_with_velocity(
        self,
        velocity_intent: t.Tensor,
        num_tokens: Optional[int] = None,
    ) -> tuple[t.Tensor, t.Tensor, int]:
        """Replan the upcoming motion conditioned on a 4-D velocity vector.

        Args:
            velocity_intent: [4] or [B, 4] tensor with channels
                `[yaw_rate_rad_s, vel_x_m_s, vel_z_m_s, hip_height_m]`. This is
                broadcast across all 4 target frames in the constraint window.
                Note Y is the gravity axis in the local-motion-rep coordinate
                frame; the lateral channel is `vel_z`, not `vel_y`. See the
                channel order assertion below.
            num_tokens: Override the model's MASKED_NUM_TOKENS sentinel. Default
                None lets the root model predict the horizon length.

        Returns:
            model_features [1, T, D], mujoco_qpos [1, T, qpos_dim], num_pred_frames.
            The internal frame buffer is updated in place; callers can ignore
            the returned tensors and just keep popping via get_next_frame().
        """
        if self.frames["mujoco_qpos"] is None:
            raise RuntimeError("NeuralPlannerCore.replan_with_velocity() called before reset()")

        velocity_intent = self._coerce_velocity_intent(velocity_intent)

        # Pull the last 4 frames of MuJoCo qpos and build the input dict the
        # legacy `_generate_inbetween_frames` consumes.
        context_mujoco_qpos = self.get_context_mujoco_qpos()
        input_state: dict = {
            "context_mujoco_qpos": context_mujoco_qpos,
            "raw_context_mujoco_qpos": context_mujoco_qpos.clone(),
            "velocity_intent": velocity_intent,
        }

        if self.FORCE_CANONICALIZATION:
            self._canonicalize_mujoco_qpos(input_state)

        # Convert canonicalized qpos to (joint_positions, joint_rotations).
        (
            input_state["context_global_joint_positions"],
            input_state["context_global_joint_rotations"],
        ) = self._converter.convert_mujoco_qpos_to_motion_transforms(
            input_state["context_mujoco_qpos"]
        )

        model_features, mujoco_qpos, num_pred_frames = self._predict_with_velocity(
            input_state, num_tokens=num_tokens
        )

        self.frames["model_features"] = model_features[:, : int(num_pred_frames), :]
        self.frames["mujoco_qpos"] = mujoco_qpos[:, : int(num_pred_frames), :]
        self._current_frame_idx = self.NUM_FRAMES_PER_TOKEN - self.PRED_OFFSETS

        return model_features, mujoco_qpos, int(num_pred_frames)

    # ------------------------------------------------------------------
    # Internal: predict + decode (forked from full_navigation_agent
    # ._generate_inbetween_frames). The differences from G1 demo:
    #   - target_local_root_values is constructed from velocity_intent, not
    #     from a target keyframe pose (no clip_holder).
    #   - target_local_poses is zero + masked off.
    #   - target_global_root_values is zero + masked off.
    # ------------------------------------------------------------------

    @t.no_grad()
    def _predict_with_velocity(
        self, input_state: dict, num_tokens: Optional[int] = None
    ) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        batch_size = 1
        MASKED_NUM_TOKENS = self._inferencer._root_model.backbone_net.MASKED_NUM_TOKENS
        fps = self._inferencer.local_motion_rep.fps
        root_joint_idx = 0
        device = self._device
        NUM_FT = self.NUM_FRAMES_PER_TOKEN

        ctx_joint_pos = input_state["context_global_joint_positions"]   # [B, 4, J, 3]
        ctx_joint_rot = input_state["context_global_joint_rotations"]   # [B, 4, J, 3, 3]
        velocity_intent = input_state["velocity_intent"]                # [B, 4]

        # ----------------------------------------------------------------
        # Past 4 frames: global + local root values from context positions.
        # Layout matches LocalRootLocalBody.compute_root_rep_from_root_pos_and_rot:
        #   channel 0 = yaw_rate (rad/s, * fps internally)
        #   channels 1-2 = vel_x, vel_z (m/s, * fps internally)
        #   channel 3 = hip height (m, world-frame Y-up)
        # ----------------------------------------------------------------
        context_global_root_pos = ctx_joint_pos[:, :, root_joint_idx, :]  # [B, 4, 3]
        context_rotation_angle = t.atan2(
            ctx_joint_rot[:, :, root_joint_idx, 0, 2],
            ctx_joint_rot[:, :, root_joint_idx, 2, 2],
        )  # [B, 4] heading angle
        context_global_root_values = t.cat(
            [
                context_global_root_pos,
                t.cos(context_rotation_angle)[..., None],
                t.sin(context_rotation_angle)[..., None],
            ],
            dim=-1,
        )  # [B, 4, 5]

        context_local_root_values = t.zeros([batch_size, NUM_FT, 4], device=device)
        # yaw-rate (channel 0) from differenced headings
        context_local_root_values[:, : NUM_FT - 1, 0] = (
            ((context_rotation_angle[:, 1:] - context_rotation_angle[:, :-1] + t.pi) % (2 * t.pi))
            - t.pi
        ) * fps
        # vel_x, vel_z (channels 1-2) from differenced xz positions in body frame
        context_local_root_values[:, : NUM_FT - 1, 1:3] = (
            context_global_root_pos[:, 1:, [0, 2]] - context_global_root_pos[:, :-1, [0, 2]]
        ) * fps
        # hip height (channel 3) directly from Y
        context_local_root_values[:, : NUM_FT - 1, 3] = (
            context_global_root_values[:, : NUM_FT - 1, 1]
        )
        # Last context frame has no t+1; copy the previous frame's velocity.
        context_local_root_values[:, NUM_FT - 1, :] = context_local_root_values[:, NUM_FT - 2, :]

        # Joint positions relative to the root, plus 6D rotations.
        joint_positions = ctx_joint_pos[:, :, 1:, :].clone()
        joint_positions[..., 0] = ctx_joint_pos[:, :, 1:, 0] - ctx_joint_pos[:, :, :1, 0]
        joint_positions[..., 2] = ctx_joint_pos[:, :, 1:, 2] - ctx_joint_pos[:, :, :1, 2]
        joint_rotation_ortho6d = matrix_to_cont6d(ctx_joint_rot)
        context_local_poses = t.cat(
            [
                joint_positions.view([batch_size, NUM_FT, -1]),
                joint_rotation_ortho6d.view([batch_size, NUM_FT, -1]),
            ],
            dim=-1,
        )

        # ----------------------------------------------------------------
        # Target 4 frames: only target_local_root_values is meaningful
        # (broadcast from velocity_intent). target_global_root_values and
        # target_local_poses are zeros + masked off (no world-frame target
        # pose, no joint-pose constraint -- the VQVAE free-samples).
        # ----------------------------------------------------------------
        target_global_root_values = t.zeros([batch_size, NUM_FT, 5], device=device)
        target_local_root_values = velocity_intent[:, None, :].expand([batch_size, NUM_FT, 4]).contiguous()
        target_local_poses = t.zeros_like(context_local_poses)

        # ----------------------------------------------------------------
        # Concatenate -> 8-frame constraint window.
        # ----------------------------------------------------------------
        local_root_values = t.cat([context_local_root_values, target_local_root_values], dim=1)
        global_root_values = t.cat([context_global_root_values, target_global_root_values], dim=1)
        local_poses = t.cat([context_local_poses, target_local_poses], dim=1)

        has_global_root_values = t.ones_like(global_root_values[:, :, 0], dtype=t.bool)
        has_local_root_values = t.ones_like(local_root_values[:, :, 0], dtype=t.bool)
        has_local_poses = t.ones_like(local_poses[:, :, 0], dtype=t.bool)
        # Last differenced velocity in the context window is invalid (no t+1
        # to differentiate against). Matches full_agent.py:457.
        has_local_root_values[:, NUM_FT - 1] = False
        # No target world-frame position constraint: let the network integrate
        # the velocity to find the end pose itself.
        has_global_root_values[:, -NUM_FT:] = False
        # No target keyframe pose: free-sample locomotion poses consistent
        # with the velocity constraint.
        has_local_poses[:, -NUM_FT:] = False

        if self.diagnostic_mask_hook is not None:
            has_global_root_values, has_local_root_values, has_local_poses = (
                self.diagnostic_mask_hook(
                    has_global_root_values,
                    has_local_root_values,
                    has_local_poses,
                    NUM_FT,
                )
            )

        if num_tokens is None:
            num_tokens_t = t.full([batch_size, 1], MASKED_NUM_TOKENS, dtype=t.int, device=device)
        else:
            num_tokens_t = t.full([batch_size, 1], int(num_tokens), dtype=t.int, device=device)

        config = {
            "num_inference_step": 1,
            "smooth_root_traj": False,
            "allow_pred_out_of_reach_num_tokens": False,
            "pose_token_sampling_use_argmax": True,
            "skip_ending_target_cond": self.SKIP_ENDING_TARGET_COND,
        }
        info: dict = {}
        pred_global_motions, num_pred_tokens = self._inferencer.predict(
            global_root_values,
            has_global_root_values,
            local_root_values,
            has_local_root_values,
            local_poses,
            has_local_poses,
            num_tokens_t,
            config=config,
            info=info,
        )

        model_features = pred_global_motions
        num_pred_frames = NUM_FT * num_pred_tokens

        mujoco_qpos = self._converter.convert_motion_features_to_mujoco_qpos(
            model_features, self._motion_rep, False
        )
        # Converter emits xyzw; flip to wxyz so the wire matches the heuristic
        # planner's convention (build_pose_payload reads root_quat_xyzw, but
        # the X2 converter has root_quat_w_first=False by default in the demo
        # path -- match that here).
        root_rot = mujoco_qpos[:, :, 3:7].clone()
        mujoco_qpos[:, :, 3:7] = root_rot[:, :, [3, 0, 1, 2]]

        if self.FORCE_CANONICALIZATION:
            input_state["mujoco_qpos"] = mujoco_qpos
            mujoco_qpos = self._uncanonicalize_mujoco_qpos(input_state)

        if self.FILTER_QPOS:
            self.frames["raw_mujoco_qpos"] = mujoco_qpos.clone()
            ctx = input_state["raw_context_mujoco_qpos"]
            num_ctx = ctx.shape[1]
            blend = t.linspace(0.3, 0.7, num_ctx)[None, :, None].to(ctx.device)
            mujoco_qpos[:, :num_ctx, :3] = (
                ctx[:, :, :3] * (1 - blend) + mujoco_qpos[:, :num_ctx, :3] * blend
            )
            mujoco_qpos[:, :num_ctx, 7:] = (
                ctx[:, :, 7:] * (1 - blend) + mujoco_qpos[:, :num_ctx, 7:] * blend
            )

        return model_features, mujoco_qpos, num_pred_frames

    # ------------------------------------------------------------------
    # Canonicalize / uncanonicalize (lifted from full_agent.py, robot-agnostic).
    # ------------------------------------------------------------------

    def _canonicalize_mujoco_qpos(self, input_state: dict) -> None:
        """In-place canonicalize the context qpos to first-frame-origin + zero-heading."""
        mujoco_qpos = input_state["context_mujoco_qpos"]
        input_state["raw_context_mujoco_qpos"] = mujoco_qpos.clone()

        first_frame_position = (
            mujoco_qpos[:, 0, :3].clone() * t.tensor([[1.0, 1.0, 0.0]]).to(mujoco_qpos.device)
        )
        first_frame_rot = quaternion_to_matrix(mujoco_qpos[:, 0, 3:7].clone())
        first_frame_heading_angle = t.atan2(first_frame_rot[:, 1, 0], first_frame_rot[:, 0, 0])
        first_frame_heading_angle[first_frame_heading_angle.isnan()] = 0.0
        first_frame_rot_heading = angle_to_Z_rotation_matrix(first_frame_heading_angle)
        inverse_first_frame_rot_heading = first_frame_rot_heading.transpose(-2, -1)

        canonicalized_root_position = t.matmul(
            inverse_first_frame_rot_heading[:, None, :, :],
            (mujoco_qpos[:, :, :3].clone() - first_frame_position)[..., None],
        )[..., 0]
        canonicalized_rot_matrix = t.matmul(
            inverse_first_frame_rot_heading[:, None, :, :],
            quaternion_to_matrix(mujoco_qpos[:, :, 3:7]),
        )
        mujoco_qpos[:, :, 3:7] = matrix_to_quaternion(canonicalized_rot_matrix)
        mujoco_qpos[:, :, :3] = canonicalized_root_position

        input_state["first_frame_heading_angle"] = first_frame_heading_angle
        input_state["first_frame_position"] = first_frame_position
        input_state["context_mujoco_qpos"] = mujoco_qpos

    def _uncanonicalize_mujoco_qpos(self, input_state: dict) -> t.Tensor:
        """Undo `_canonicalize_mujoco_qpos`. Returns the world-frame qpos."""
        mujoco_qpos = input_state["mujoco_qpos"]
        first_frame_heading_angle = input_state["first_frame_heading_angle"]
        first_frame_position = input_state["first_frame_position"]

        first_frame_rot_heading = angle_to_Z_rotation_matrix(first_frame_heading_angle)
        current_first_frame_rotation = quaternion_to_matrix(mujoco_qpos[:, :1, 3:7])
        current_first_frame_heading_angle = t.atan2(
            current_first_frame_rotation[:, :, 1, 0],
            current_first_frame_rotation[:, :, 0, 0],
        )
        current_first_frame_rot_heading = angle_to_Z_rotation_matrix(current_first_frame_heading_angle)
        rot_matrix = quaternion_to_matrix(mujoco_qpos[:, :, 3:7])
        rot_matrix = t.matmul(
            first_frame_rot_heading[:, None, :, :],
            t.matmul(current_first_frame_rot_heading.transpose(-2, -1), rot_matrix),
        )
        root_positions = t.matmul(
            first_frame_rot_heading[:, None, :, :],
            t.matmul(
                current_first_frame_rot_heading.transpose(-2, -1),
                mujoco_qpos[:, :, :3, None],
            ),
        )[..., 0]
        root_positions = (
            root_positions
            - root_positions[:, :1, :] * t.tensor([[[1.0, 1.0, 0.0]]]).to(mujoco_qpos.device)
            + first_frame_position
        )
        mujoco_qpos[:, :, 3:7] = matrix_to_quaternion(rot_matrix)
        mujoco_qpos[:, :, :3] = root_positions
        return mujoco_qpos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce_velocity_intent(self, velocity_intent: t.Tensor) -> t.Tensor:
        """Accept [4] or [B, 4]; return [B, 4] on self._device with float dtype."""
        if not isinstance(velocity_intent, t.Tensor):
            velocity_intent = t.as_tensor(velocity_intent, dtype=t.float32)
        if velocity_intent.ndim == 1:
            velocity_intent = velocity_intent[None, :]
        if velocity_intent.shape[-1] != 4:
            raise ValueError(
                f"velocity_intent last-dim must be 4 (yaw_rate, vel_x, vel_z, hip_h); "
                f"got shape {tuple(velocity_intent.shape)}"
            )
        return velocity_intent.to(self._device).float()
