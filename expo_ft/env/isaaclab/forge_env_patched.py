"""FORGE environment, patched for a pretrained pi05_droid_jointpos rollout.

Two changes relative to stock FORGE (isaaclab_tasks.direct.forge.forge_env.ForgeEnv),
both confined to this file:

1. Cameras. FORGE ships no visual sensors (it's state-based). We add the
   exterior + wrist TiledCameras whose poses were validated visually
   (see /mnt/scratch/forge_camera_check or the task YAML's `cameras` block).

2. Action interface. Stock FORGE's `_apply_action` consumes a 7D task-space
   vector (pos delta 3 + rot delta 3 [yaw survives] + success-pred 1) and
   drives a hand-written task-space impedance controller. pi05_droid_jointpos
   outputs something structurally different: an 8D vector — a joint-position
   DELTA for the 7 arm joints (relative to the current joint state, confirmed
   against multiple independent sources — NOT already absolute) plus an
   absolute gripper target. There is no configuration flag in FORGE to accept
   this; `_apply_action` is overridden here instead of built as a translation
   layer on top of the original, because the two are physically different
   quantities (Cartesian pose vs. joint position) with no lossless conversion
   between them that preserves the pretraining distribution.

   The replacement is intentionally NOT a hand-derived impedance law: FORGE's
   own zeroed-out arm actuator gains (stiffness=damping=0.0 in
   isaaclab_tasks/direct/factory/factory_env_cfg.py) are overridden back to
   Isaac Lab's own defaults (80.0 / 4.0) at config-build time in
   forge_server.py, so PhysX's already-existing implicit PD actuator does the
   compliant tracking — exactly what ManiSkill's `pd_joint_delta_pos` control
   mode already provides elsewhere in this project. `_apply_action` here is
   just: current + delta -> set_joint_position_target.

   `_pre_physics_step` is also overridden to skip FORGE's own EMA action
   smoothing (`ema_factor`), which was tuned for the task-space action's
   scale/semantics and has no established meaning for a joint-position delta.

TODO (flagged, not blocking): gripper normalization. The 8th action
dimension is passed through as a joint-position target for both finger
joints, linearly mapped from the assumed DROID convention (0 = open,
1 = closed) to FORGE's own finger-open range (0.0 = closed, ~0.04 = open,
see `factory_env_cfg.py`'s `robot.init_state`). This mapping is a best
effort, not independently verified — confirm it visually on the first live
rollout (does the gripper open/close in the expected direction?) and adjust
GRIPPER_OPEN_WIDTH / the sign below if not.
"""
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

# Validated camera poses (see the tactile-phase camera check). Kept as module
# constants rather than cfg fields so this file has no dependency beyond the
# task YAML's num_envs=1 assumption; override via the CFG_OVERRIDES dict
# below from forge_server.py if a task ever needs different poses.
EXTERIOR_EYE = (1.05, 0.35, 0.35)
EXTERIOR_TARGET = (0.57, 0.02, 0.08)
EXTERIOR_FOCAL = 24.0
WRIST_POS = (0.05, 0.0, 0.0)
WRIST_ROT = (0.6916, -0.1471, -0.1471, 0.6916)  # ROS convention, on panda_hand
WRIST_FOCAL = 18.0
CAMERA_SIZE = 224

GRIPPER_OPEN_WIDTH = 0.04  # meters per finger, see factory_env_cfg.py robot.init_state


def _look_at_quat_opengl(eye, target):
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
    eye_t = torch.tensor([eye], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    rot = create_rotation_matrix_from_view(eye_t, target_t, up_axis="Z")
    return tuple(quat_from_matrix(rot)[0].tolist())


class ForgeEnvJointPosPi05(ForgeEnv):
    """FORGE + cameras + 8D joint-position-delta action interface."""

    def _setup_scene(self):
        super()._setup_scene()

        ext_rot = _look_at_quat_opengl(EXTERIOR_EYE, EXTERIOR_TARGET)
        self._exterior_cam = TiledCamera(TiledCameraCfg(
            prim_path="/World/envs/env_.*/ExteriorCam",
            offset=TiledCameraCfg.OffsetCfg(pos=EXTERIOR_EYE, rot=ext_rot, convention="opengl"),
            data_types=["rgb"], width=CAMERA_SIZE, height=CAMERA_SIZE,
            spawn=sim_utils.PinholeCameraCfg(focal_length=EXTERIOR_FOCAL, clipping_range=(0.01, 10.0)),
            update_latest_camera_pose=True,
        ))
        self._wrist_cam = TiledCamera(TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/panda_hand/WristCam",
            offset=TiledCameraCfg.OffsetCfg(pos=WRIST_POS, rot=WRIST_ROT, convention="ros"),
            data_types=["rgb"], width=CAMERA_SIZE, height=CAMERA_SIZE,
            spawn=sim_utils.PinholeCameraCfg(focal_length=WRIST_FOCAL, clipping_range=(0.01, 10.0)),
            update_latest_camera_pose=True,
        ))
        self.scene.sensors["exterior_cam"] = self._exterior_cam
        self.scene.sensors["wrist_cam"] = self._wrist_cam

    def _pre_physics_step(self, action):
        """Compute and cache the joint-position TARGET once per env.step()
        (this is called once), not once per decimation substep. This is the
        fix for the bug found in testing: _apply_action runs `decimation`
        times (8) per env.step() (confirmed in isaaclab's DirectRLEnv.step()),
        and the original version recomputed `self.joint_pos + delta_arm`
        fresh on every one of those calls using the LIVE, already-moving
        joint position — so the delta kept getting re-added on top of
        wherever the arm had already drifted to mid-step, instead of being
        applied once. FORGE's own original _apply_action never hits this
        because it targets a essentially-static reference frame (the fixed
        asset's observed pose, not the live end-effector position); our
        joint-delta action has no such stable frame to lean on, so the
        target has to be captured explicitly, once, before the decimation
        loop starts — exactly like ManiSkill's pd_joint_delta_pos does.

        No EMA smoothing of the action itself (see module docstring).
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)
        self.actions = action.clone().to(self.device)

        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        delta_arm = self.actions[:, 0:7]
        gripper_cmd = self.actions[:, 7]  # assumed DROID convention: 0=open, 1=closed

        target = self.ctrl_target_joint_pos.clone()
        target[:, 0:7] = self.joint_pos[:, 0:7] + delta_arm  # current pos read ONCE, here
        finger_width = (1.0 - gripper_cmd).clamp(0.0, 1.0) * GRIPPER_OPEN_WIDTH
        target[:, 7:9] = finger_width.unsqueeze(-1)

        self.ctrl_target_joint_pos[:] = target
        self._target_joint_pos = target  # held fixed across all decimation substeps

    def _apply_action(self):
        """Apply the FIXED target computed once in _pre_physics_step — no
        recomputation here, deliberately (see that method's docstring)."""
        # FORGE's own _apply_action (which we fully replace) sets these two —
        # they feed only its dense reward's action penalty (_get_rewards,
        # pos_error = norm(self.delta_pos)), which we don't use (sparse,
        # ground-truth success instead). Isaac Lab's step() calls
        # _get_rewards() unconditionally regardless, so these must exist to
        # avoid an AttributeError even though their value is never read by us.
        self.delta_pos = torch.zeros_like(self.fingertip_midpoint_pos)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)

        self._robot.set_joint_position_target(self._target_joint_pos)
        # FORGE's own _reset_idx (inherited unmodified) calls
        # close_gripper_in_place() in a 0.25s settling loop right after reset,
        # which computes torques via FORGE's task-space controller and applies
        # them with set_joint_effort_target() on the ARM DOFs — not just the
        # gripper, despite the name. That call is sticky: whatever value it
        # last set stays applied every physics step until overwritten, and we
        # never touch effort target ourselves (position control only), so the
        # last reset-time torque silently persisted for the entire episode —
        # confirmed as the cause of the constant (non-decaying) drift measured
        # on joint[0]/joint[2] in testing. Explicitly zeroed every step so no
        # stray effort command can ever linger, regardless of what reset does.
        self._robot.set_joint_effort_target(torch.zeros_like(self._target_joint_pos))
