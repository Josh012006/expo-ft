"""FORGE environment, patched for a pretrained pi05_droid_jointpos rollout.

Two changes relative to stock FORGE (isaaclab_tasks.direct.forge.forge_env.ForgeEnv),
both confined to this file:

1. Cameras. FORGE ships no visual sensors (it's state-based). We add the
   exterior + wrist TiledCameras whose poses were validated visually
   (see /mnt/scratch/forge_camera_check or the task YAML's `cameras` block).

2. Action interface. Stock FORGE's `_apply_action` consumes a 7D task-space
   vector (pos delta 3 + rot delta 3 [yaw survives] + success-pred 1) and
   drives a hand-written task-space impedance controller. pi05_droid_jointpos
   outputs something structurally different: an 8D vector — 7 ABSOLUTE
   joint-position targets plus a gripper command (0 = open, 1 = closed).
   (The raw network output is a chunk of OFFSETS from the joint state at plan
   time — openpi's DeltaActions encoding; the client adds that state back, the
   equivalent of openpi's AbsoluteActions: scripts/eval_policy.py,
   `chunk_action_reference`. What reaches this server is therefore already
   absolute, and nothing here depends on the live joint position.)
   There is no configuration flag in FORGE to accept
   this; `_apply_action` is overridden here instead of built as a translation
   layer on top of the original, because the two are physically different
   quantities (Cartesian pose vs. joint position) with no lossless conversion
   between them that preserves the pretraining distribution.

   The replacement is intentionally NOT a hand-derived impedance law: FORGE's
   own zeroed-out arm actuator gains (stiffness=damping=0.0 in
   isaaclab_tasks/direct/factory/factory_env_cfg.py) are overridden back to
   Isaac Lab's own defaults (80.0 / 4.0) at config-build time in
   forge_server.py, so PhysX's already-existing implicit PD actuator does the
   compliant tracking — the same PD joint tracking ManiSkill's pd_joint_*
   controllers provide elsewhere in this project. `_apply_action` here is
   just: absolute target -> set_joint_position_target.

   `_pre_physics_step` is also overridden to skip FORGE's own EMA action
   smoothing (`ema_factor`), which was tuned for the task-space action's
   scale/semantics and has no established meaning for a joint-position target.

3. Robot and taxels (expo_ft/env/isaaclab/robots.py, YAML field `robot`). With the Robotiq 2F-85 profiles the gripper is ONE
   driven joint (`finger_joint`, radians, 0 = open) plus five mimic joints that PhysX moves by itself, instead of
   FORGE's two prismatic fingers. FORGE keeps writing "width in m per finger, 0 = closed, GRIPPER_OPEN_WIDTH = open" in
   columns 7:9 of its joint tensors; `_install_gripper_adapter` converts that width at the two places where a joint
   tensor goes to the simulation (set_joint_position_target, write_joint_state_to_sim), so FORGE, this file and the
   server keep one gripper convention whatever the robot. `gripper_width()` is the inverse (what the policy sees).
   With `robotiq_2f85_tsf` two ContactSensors read the 7 x 4 tactile zones of each finger: `tactile_grids()`.

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
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from expo_ft.env.isaaclab.robots import TACTILE_COLS, TACTILE_ROWS, get_profile

# Validated camera poses (see the tactile-phase camera check). Kept as module
# constants rather than cfg fields so this file has no dependency beyond the
# task YAML's num_envs=1 assumption; override via the CFG_OVERRIDES dict
# below from forge_server.py if a task ever needs different poses.
EXTERIOR_EYE = (1.05, 0.35, 0.35)
EXTERIOR_TARGET = (0.57, 0.02, 0.08)
EXTERIOR_FOCAL = 24.0
CAMERA_SIZE = 224
# The wrist camera (mount, pose, focal length) comes from the robot profile (robots.py): for "franka" it is exactly the
# camera that was validated here before profiles existed (on panda_hand, pos (0.05, 0, 0), ROS quaternion
# (0.6916, -0.1471, -0.1471, 0.6916), focal 18).

GRIPPER_OPEN_WIDTH = 0.04  # meters per finger, see factory_env_cfg.py robot.init_state


def _look_at_quat_opengl(eye, target):
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
    eye_t = torch.tensor([eye], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    rot = create_rotation_matrix_from_view(eye_t, target_t, up_axis="Z")
    return tuple(quat_from_matrix(rot)[0].tolist())


class ForgeEnvJointPosPi05(ForgeEnv):
    """FORGE + cameras + 8D action interface: 7 absolute joint targets + gripper."""

    def __init__(self, cfg, render_mode=None, gripper_can_open: bool = True, robot_profile=None, **kwargs):
        """gripper_can_open: when False, the policy's gripper channel
        (action[:, 7]) is ignored every step and the gripper target is left
        at whatever it already was — which, since nothing else ever writes
        it in that case, stays at FORGE's own reset-time grasp width
        (close_gripper_in_place(), factory_env.py) for the entire episode.
        Added because the pretrained checkpoint's zero-shot gripper output
        tends to open the gripper early in an episode (plausible cause:
        DROID demonstrations overwhelmingly start reaching toward an object
        with the gripper open, not already holding one — an episode-start
        condition FORGE's own task never matches) — dropping the held peg
        before any insertion attempt is possible. This is a blunt, temporary
        workaround, not a fix to that distribution mismatch itself."""
        self._gripper_can_open = gripper_can_open
        self._profile = robot_profile if robot_profile is not None else get_profile("franka")
        self._is_robotiq = self._profile.gripper == "robotiq_2f85"
        self._tactile_sensors = {}          # "left" / "right" -> ContactSensor (profiles with taxels only)
        self._zone_sync = None              # fallback, see _init_zone_sync
        self._finger_adapter = None         # Robotiq only, see _install_gripper_adapter
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        if self._tactile_sensors:
            self._init_tactile()
            self._sync_zones()
        if self._is_robotiq:
            self._install_gripper_adapter()

    # ------------------------------------------------------------------ gripper (robot profile)

    def _theta_to_width(self, theta):
        """Robotiq finger joint angle (rad, 0 = open) -> FORGE's width per finger (m, 0 = closed, GRIPPER_OPEN_WIDTH = open)."""
        return GRIPPER_OPEN_WIDTH * (1.0 - theta / self._finger_adapter["theta_max"]).clamp(0.0, 1.0)

    def _compute_intermediate_values(self, dt):
        """FORGE's own, then (Robotiq) the gripper columns of FORGE's joint_pos copy are put in FORGE's convention.

        FORGE writes self.joint_pos back to the simulation (its reset IK loop) and builds its gripper targets from it:
        every joint tensor on FORGE's side must therefore hold "width per finger" in columns 7: (what the Franka Hand's
        prismatic fingers are), and only the two robot methods wrapped in _install_gripper_adapter see real angles."""
        super()._compute_intermediate_values(dt)
        if self._finger_adapter is not None:
            self.joint_pos[:, 7:] = self._theta_to_width(self.joint_pos[:, 7]).unsqueeze(-1)

    def gripper_width(self, fresh: bool = False):
        """(num_envs,) finger opening in FORGE's convention: m per finger, 0 = closed, GRIPPER_OPEN_WIDTH = fully open,
        whatever the gripper (for the Robotiq: the linear image of the finger joint angle, 0 rad = open).
        fresh=False reads FORGE's own joint_pos (refreshed by its observation code, what the policy was given until
        now); fresh=True reads the simulation's current state."""
        if self._finger_adapter is not None:
            return self._theta_to_width(self._robot.data.joint_pos[:, 7]) if fresh else self.joint_pos[:, 7]
        return (self._robot.data.joint_pos if fresh else self.joint_pos)[:, 7]

    def gripper_effort(self):
        """(num_envs,) torque (N.m) that the finger joint's PD law demanded at the last physics sub-step, clipped to its effort
        limit (Isaac Lab's applied_torque: the COMMAND of the motor, not a measured joint torque). Robotiq only (the Franka
        Hand's fingers are prismatic: a force, in another unit); None for the Franka Hand."""
        return self._robot.data.applied_torque[:, 7] if self._is_robotiq else None

    def gripper_info(self):
        """What the adapter measured / uses (debug op `robot_info` of the server); {} for the Franka Hand."""
        if self._finger_adapter is None:
            return {}
        a = self._finger_adapter
        return {"theta_max_rad": a["theta_max"], "mimic_joints": a["names"], "mimic_slope": a["slope"].tolist(),
                "mimic_offset": a["offset"].tolist(), "calibration": a["calibration"],
                "zones_are_articulation_links": self._zones_are_links(),
                "zone_sync_fallback": self._zone_sync is not None}

    def _install_gripper_adapter(self):
        """Robotiq 2F-85: make FORGE's "width per finger" columns drive the single finger joint + its mimic joints.

        Joint order of NVIDIA's USD (verified with scripts/isaaclab/inspect_robot_usd.py --spawn): 0-6 panda_joint1-7,
        7 finger_joint (the only driven gripper joint, 0 = open .. upper limit = closed), 8-12 mimic joints. FORGE
        writes the gripper as width w in columns 7:9 (target) and 7: (reset state). The two robot methods below
        replace column 7 by theta = theta_max * (1 - w / GRIPPER_OPEN_WIDTH) and the mimic columns by
        slope * theta + offset, measured here in free air (a mimic joint is a PhysX constraint: a reset state that
        disagrees with it is fought by the solver). Only full-width tensors (num_joints columns, no joint_ids)
        are converted: that is the only way FORGE and this file call them.
        """
        robot = self._robot
        names = list(robot.data.joint_names)
        if names.index("finger_joint") != 7:
            raise RuntimeError(f"finger_joint is not joint 7 in this USD: {names}")
        mimic = ["right_outer_knuckle_joint", "left_inner_finger_joint", "right_inner_finger_joint",
                 "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"]
        ids = [names.index(n) for n in mimic]
        theta_max = float(robot.data.joint_pos_limits[0, 7, 1])
        adapter = {"theta_max": theta_max, "names": mimic, "ids": ids,
                   "slope": torch.zeros(len(ids), device=self.device), "offset": torch.zeros(len(ids), device=self.device)}
        adapter["calibration"] = self._calibrate_mimic(adapter)
        self._finger_adapter = adapter

        raw_set_target = robot.set_joint_position_target
        raw_write_state = robot.write_joint_state_to_sim
        num_joints = robot.num_joints

        def convert(q):
            if q.dim() != 2 or q.shape[-1] != num_joints:
                return q
            q = q.clone()
            theta = theta_max * (1.0 - q[:, 7] / GRIPPER_OPEN_WIDTH).clamp(0.0, 1.0)
            q[:, 7] = theta
            q[:, ids] = theta.unsqueeze(-1) * adapter["slope"] + adapter["offset"]
            return q

        def set_target(target, joint_ids=None, env_ids=None):
            return raw_set_target(target if joint_ids is not None else convert(target), joint_ids=joint_ids, env_ids=env_ids)

        def write_state(position, velocity, joint_ids=None, env_ids=None):
            out = raw_write_state(position if joint_ids is not None else convert(position), velocity,
                                  joint_ids=joint_ids, env_ids=env_ids)
            self._sync_zones()
            return out

        robot.set_joint_position_target = set_target          # instance attributes: only this robot is affected
        robot.write_joint_state_to_sim = write_state
        print(f"[forge_env] Robotiq adapter: theta_max={theta_max:.4f} rad, mimic slope={adapter['slope'].tolist()} "
              f"offset={adapter['offset'].tolist()}", flush=True)

    def _calibrate_mimic(self, adapter, steps=60, max_steps=400):
        """Mimic joint angle = slope * finger_joint angle + offset, measured: drive the finger joint to 0 and to half its
        range in free air, wait until it rests, read all joints. Leaves the robot as found (joint state restored)."""
        robot, ids, theta_max = self._robot, adapter["ids"], adapter["theta_max"]
        q_start, v_start = robot.data.joint_pos.clone(), torch.zeros_like(robot.data.joint_vel)
        points = []
        for frac in (0.0, 0.5):
            target = q_start.clone()
            target[:, 7] = frac * theta_max
            for i in range(max_steps):
                robot.set_joint_position_target(target)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)
                if i + 1 >= steps and float(robot.data.joint_vel[:, 7:].abs().max()) < 0.02:
                    break
            points.append(robot.data.joint_pos[0].clone())
        robot.write_joint_state_to_sim(q_start, v_start)
        self._sync_zones()
        robot.set_joint_position_target(q_start)
        robot.reset()
        dtheta = points[1][7] - points[0][7]
        slope = (points[1][ids] - points[0][ids]) / dtheta
        offset = points[0][ids] - slope * points[0][7]
        # Gearing of the mimic joints in NVIDIA's USD is +-1 with no offset: anything else means the fingers hit
        # something during the measurement, then the authored values are used (signs from the measurement).
        bad = bool((torch.abs(dtheta) < 0.2 * theta_max) or (torch.abs(slope.abs() - 1.0) > 0.15).any()
                   or (offset.abs() > 0.05).any())
        if bad:
            print(f"[forge_env] WARNING mimic calibration looks wrong (slope {slope.tolist()}, offset {offset.tolist()}); "
                  "using slope = +-1, offset = 0", flush=True)
            slope, offset = torch.sign(slope), torch.zeros_like(offset)
        adapter["slope"], adapter["offset"] = slope, offset
        return {"measured_dtheta_rad": float(dtheta), "fallback": bad}

    # ------------------------------------------------------------------ taxels (robot profile with tactile)

    def tactile_grids(self):
        """(num_envs, 2, TACTILE_ROWS, TACTILE_COLS) net contact force (N) of each zone, [:, 0] left finger, [:, 1] right.
        Mean over the `decimation` physics sub-steps of the last env step of the norm of the force of each zone; row 0 =
        fingertip end. Zones are rigid bodies with a contact report: this is the contact force the solver applied to
        them (normal + friction), not a calibrated taxel reading."""
        out = []
        for side in ("left", "right"):
            sensor = self._tactile_sensors[side]
            force = torch.linalg.norm(sensor.data.net_forces_w_history, dim=-1).mean(dim=1)    # (N, bodies)
            out.append(force[:, self._tactile_order[side]].reshape(self.num_envs, TACTILE_ROWS, TACTILE_COLS))
        return torch.stack(out, dim=1)

    def _init_tactile(self):
        import re
        self._tactile_order = {}
        for side, sensor in self._tactile_sensors.items():
            rc = [tuple(int(v) for v in re.fullmatch(r"zone_r(\d+)_c(\d+)", n).groups()) for n in sensor.body_names]
            want = [(r, c) for r in range(TACTILE_ROWS) for c in range(TACTILE_COLS)]
            if sorted(rc) != want:
                raise RuntimeError(f"tactile sensor {side}: zones {sensor.body_names} are not the {TACTILE_ROWS}x{TACTILE_COLS} grid")
            self._tactile_order[side] = torch.tensor([rc.index(w) for w in want], device=self.device)
        if not self._zones_are_links():
            self._init_zone_sync()
        print(f"[forge_env] tactile: {TACTILE_ROWS}x{TACTILE_COLS} zones per finger, zones are articulation links: "
              f"{self._zones_are_links()}, sync fallback: {self._zone_sync is not None}", flush=True)

    def _zones_are_links(self):
        return any(n.startswith("zone_r") for n in self._robot.body_names)

    def _init_zone_sync(self):
        """FALLBACK, only if PhysX did not make the zones links of the articulation (tactile_exclude_zone_joints, or a
        link limit): then they are free bodies held to the fingers by fixed joints, which write_joint_state_to_sim
        (every reset) does not move: the joints would drag them over 0.4 m through the table and the peg at the next
        step. After each teleport of the arm the zones are put where their joints want them."""
        import re
        from isaacsim.core.simulation_manager import SimulationManager
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics
        stage, sim_view = get_current_stage(), SimulationManager.get_physics_sim_view()
        names = list(self._robot.body_names)
        self._zone_sync = []
        for side in ("left", "right"):
            view = sim_view.create_rigid_body_view(f"/World/envs/env_*/Robot/TactileZones/{side}/zone_r*_c*")
            offsets, envs = [], []
            for path in view.prim_paths:
                joint = UsdPhysics.Joint(stage.GetPrimAtPath(path + "/fixed_joint"))
                offsets.append(list(joint.GetLocalPos0Attr().Get()))
                envs.append(int(re.search(r"/env_(\d+)/", path).group(1)))
            self._zone_sync.append({
                "view": view, "off": torch.tensor(offsets, dtype=torch.float32, device=self.device),
                "env": torch.tensor(envs, device=self.device), "finger": names.index(f"{side}_inner_finger"),
                "idx": torch.arange(view.count, dtype=torch.int32, device=self.device)})

    def _sync_zones(self):
        if not self._zone_sync:
            return
        try:
            from isaaclab.utils.math import quat_apply
            from isaacsim.core.simulation_manager import SimulationManager
            SimulationManager.get_physics_sim_view().update_articulations_kinematic()
            links = self._robot.root_physx_view.get_link_transforms()               # (N, bodies, 7), x y z qx qy qz qw
            for z in self._zone_sync:
                tf = links[z["env"], z["finger"]]
                pos = tf[:, :3] + quat_apply(tf[:, [6, 3, 4, 5]], z["off"])
                z["view"].set_transforms(torch.cat([pos, tf[:, 3:7]], dim=-1).contiguous(), z["idx"])
                z["view"].set_velocities(torch.zeros(len(pos), 6, device=self.device), z["idx"])
        except Exception as e:        # best effort: report once, then stop trying
            print(f"[forge_env] WARNING zone sync failed ({e!r}); disabled", flush=True)
            self._zone_sync = None

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
        wrist = self._profile.wrist_camera
        wrist_spawn = dict(focal_length=wrist.focal_length, clipping_range=(0.01, 10.0))
        if wrist.horizontal_aperture is not None:
            wrist_spawn["horizontal_aperture"] = wrist.horizontal_aperture
        self._wrist_cam = TiledCamera(TiledCameraCfg(
            prim_path=f"/World/envs/env_.*/Robot/{wrist.mount}/WristCam",
            offset=TiledCameraCfg.OffsetCfg(pos=wrist.pos, rot=wrist.rot, convention=wrist.convention),
            data_types=["rgb"], width=CAMERA_SIZE, height=CAMERA_SIZE,
            spawn=sim_utils.PinholeCameraCfg(**wrist_spawn),
            update_latest_camera_pose=True,
        ))
        self.scene.sensors["exterior_cam"] = self._exterior_cam
        self.scene.sensors["wrist_cam"] = self._wrist_cam

        if self._profile.tactile:
            # one sensor per finger: Isaac Lab's ContactSensor takes its bodies as the direct children of ONE prim.
            # history_length = decimation: env.step() updates the scene after every physics sub-step, so the buffer
            # holds the forces of the sub-steps of the last env step.
            for side in ("left", "right"):
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/TactileZones/{side}/zone_r.*_c.*",
                    history_length=int(self.cfg.decimation), update_period=0.0, debug_vis=False))
                self._tactile_sensors[side] = sensor
                self.scene.sensors[f"tactile_{side}"] = sensor

    def _pre_physics_step(self, action):
        """Cache the joint-position TARGET once per env.step().

        _apply_action runs `decimation` (8) times per env.step() (isaaclab's
        DirectRLEnv.step()); the target is computed here, once, and held fixed
        across those sub-steps. With absolute targets it does not depend on the
        live joint position at all, so there is nothing to drift between
        sub-steps (an earlier version added a per-step increment to the live
        position on every sub-step and compounded it).

        No EMA smoothing of the action itself (see module docstring).
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)
        self.actions = action.clone().to(self.device)

        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        arm_target = self.actions[:, 0:7]     # absolute joint targets (rad)
        gripper_cmd = self.actions[:, 7]      # DROID convention: 0=open, 1=closed

        target = self.ctrl_target_joint_pos.clone()
        target[:, 0:7] = arm_target           # used as-is: no dependence on the live position
        if self._gripper_can_open:
            finger_width = (1.0 - gripper_cmd).clamp(0.0, 1.0) * GRIPPER_OPEN_WIDTH
            target[:, 7:9] = finger_width.unsqueeze(-1)
        # else: target[:, 7:9] left as cloned from self.ctrl_target_joint_pos,
        # i.e. whatever it already was — the policy's gripper output is never
        # read into the command at all.

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
