"""Visual check of exterior + wrist cameras on a FORGE env. Run with .venv-isaaclab.

Exterior camera: its pose is baked into the camera config at spawn time (look-at
computed from --eye/--target, env frame, same idea as ManiSkill's look_at), instead
of being moved at runtime -- runtime set_world_poses_from_view() had no visible
effect in this env.
Wrist camera: mounted on panda_hand with a tunable offset (ROS convention: +Z forward).
Verified: panda_hand's +Z points down toward the fingers in franka_mimic.usd, so an
identity rotation looks straight down; vertical edges (the peg) then project as radial
lines, which is why the peg looked "horizontal" in earlier images.
update_latest_camera_pose=True on both, so printed poses are the REAL current ones
(the default False only reports the spawn-time pose).
Also prints the wrist camera's forward axis and panda_hand's axes in world frame, to
find which hand axis actually points toward the fingers.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Forge-PegInsert-Direct-v0")
parser.add_argument("--out", default="forge_camera_check")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--size", type=int, default=224)
parser.add_argument("--eye", type=float, nargs=3, default=[1.05, 0.35, 0.35],
                    help="Exterior camera position, env frame.")
parser.add_argument("--target", type=float, nargs=3, default=[0.57, 0.02, 0.08],
                    help="Exterior camera look-at point, env frame (fixed asset is ~[0.57-0.63, 0.02-0.04, 0.0-0.06]).")
parser.add_argument("--focal", type=float, default=24.0)
parser.add_argument("--wrist-pos", type=float, nargs=3, default=[0.05, 0.0, 0.0])
parser.add_argument("--wrist-rot", type=float, nargs=4, default=[0.6916, -0.1471, -0.1471, 0.6916],
                    help="Quaternion (w, x, y, z), ROS convention. Default = ~24 deg tilt toward the fingers "
                         "(about camera Y, q=(0.978, 0, -0.208, 0)) composed with a +90 deg roll about the "
                         "optical axis, so the gripper sits at the BOTTOM of the image like DROID wrist views. "
                         "Tilt only (gripper on the left): 0.978 0 -0.208 0.")
parser.add_argument("--wrist-focal", type=float, default=18.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import os
import imageio.v3 as iio
import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_apply, quat_from_matrix
import isaaclab_tasks  # noqa: F401 (registers envs)
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

SIZE = args.size


def look_at_quat_opengl(eye, target):
    """(w, x, y, z) quaternion, OpenGL camera convention, looking from eye to target (Z up)."""
    eye_t = torch.tensor([eye], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    rot = create_rotation_matrix_from_view(eye_t, target_t, up_axis="Z")
    return tuple(quat_from_matrix(rot)[0].tolist())


EXT_ROT = look_at_quat_opengl(args.eye, args.target)
print("exterior cam quat (opengl, w,x,y,z):", [round(v, 4) for v in EXT_ROT])


class ForgeEnvWithCameras(ForgeEnv):
    def _setup_scene(self):
        super()._setup_scene()
        self._ext_cam = TiledCamera(TiledCameraCfg(
            prim_path="/World/envs/env_.*/ExteriorCam",
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(args.eye), rot=EXT_ROT, convention="opengl"),
            data_types=["rgb"], width=SIZE, height=SIZE,
            spawn=sim_utils.PinholeCameraCfg(focal_length=args.focal, clipping_range=(0.01, 10.0)),
            update_latest_camera_pose=True,
        ))
        self._wrist_cam = TiledCamera(TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/panda_hand/WristCam",
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(args.wrist_pos), rot=tuple(args.wrist_rot), convention="ros"
            ),
            data_types=["rgb"], width=SIZE, height=SIZE,
            spawn=sim_utils.PinholeCameraCfg(focal_length=args.wrist_focal, clipping_range=(0.01, 10.0)),
            update_latest_camera_pose=True,
        ))
        self.scene.sensors["exterior_cam"] = self._ext_cam
        self.scene.sensors["wrist_cam"] = self._wrist_cam


def r(x):
    return [round(v, 3) for v in x.tolist()]


def dump_poses(env, tag):
    dev = env.device
    print(f"--- poses ({tag}) ---")
    print("env_origin      :", r(env.scene.env_origins[0]))
    print("ext_cam pos_w   :", r(env._ext_cam.data.pos_w[0]))
    print("wrist_cam pos_w :", r(env._wrist_cam.data.pos_w[0]))
    print("fixed_pos (env) :", r(env.fixed_pos[0]))
    print("held_pos  (env) :", r(env.held_pos[0]))
    print("fingertip (env) :", r(env.fingertip_midpoint_pos[0]))

    # Wrist camera optical axis (ROS: +Z forward) expressed in world frame.
    fwd = quat_apply(env._wrist_cam.data.quat_w_ros[:1], torch.tensor([[0.0, 0.0, 1.0]], device=dev))
    print("wrist cam forward (world):", r(fwd[0]))

    # panda_hand axes in world frame -- the one close to (0, 0, -1) points toward the fingers.
    hand_idx = env._robot.body_names.index("panda_hand")
    hq = env._robot.data.body_quat_w[:1, hand_idx]
    axes = {}
    for name, v in (("x", [1.0, 0.0, 0.0]), ("y", [0.0, 1.0, 0.0]), ("z", [0.0, 0.0, 1.0])):
        axes[name] = r(quat_apply(hq, torch.tensor([v], device=dev, dtype=torch.float32))[0])
    print("hand axes (world)        :", axes)


env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
env_cfg.sim.render_interval = env_cfg.decimation  # one render per env step
env = ForgeEnvWithCameras(cfg=env_cfg, render_mode=None)
env.reset()

zero = torch.zeros((1, env_cfg.action_space), device=env.device)
env.step(zero)
dump_poses(env, "after reset")

os.makedirs(args.out, exist_ok=True)
for t in range(args.steps):
    env.step(zero)
    if t in (0, args.steps - 1):
        for name, cam in (("exterior", env._ext_cam), ("wrist", env._wrist_cam)):
            img = cam.data.output["rgb"][0].cpu().numpy()
            iio.imwrite(f"{args.out}/{name}_step{t:03d}.png", img)
            print(f"saved {name} step {t}: shape={img.shape} min={img.min()} max={img.max()}")

dump_poses(env, "end")
env.close()
app.close()
