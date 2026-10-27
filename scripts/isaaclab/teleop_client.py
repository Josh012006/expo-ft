
"""Teleoperation client for the FORGE PegInsert simulation -- STEP 3/3: you drive the arm and record demos.

Runs on YOUR PC (Windows / WSL is fine), not on the cluster. The simulation keeps running in forge_server.py on an
L40S node; this script connects to it through an SSH tunnel and shows what the policy sees:

    left   exterior camera (224x224, enlarged)
    right  wrist camera    (224x224, enlarged)
    under  the offset peg -> hole (ground truth, simulation only), the command you sent, the contact force,
           the state of the inputs and the time of every round trip to the server

You move the PEG (it is rigidly held: the fingertip follows) with the mouse and the keyboard. The orientation of
the hand is held fixed and the gripper stays closed, exactly as in the scripted demos. Each control step the
server turns your move into 7 joint targets (FORGE's own IK), applies them and answers with the new observation:
one round trip per step ("teleop_step" of forge_server.py -- restart the server after pulling this version).

SLOW MOTION. The simulation only advances when you step it, and one step is always 1/15 s of simulated time, however
long it takes on the wall clock (~0.3-0.4 s through the tunnel, the simulation alone needs ~0.18 s). By default the
simulation steps ONLY when you give an input (mouse move, W/S or Space), so you can stop and look at no cost. What
is recorded is therefore at the policy's 15 Hz, in the same format as the scripted demos. C switches to AUTO mode,
where the simulation steps at every loop even without input (needed to let the 5 "inserted" steps pass once the peg
is in the hole).

RECORDING. Every step is kept in memory: the observation the move was computed from, and the 8 values sent to the arm
(7 absolute joint targets + gripper 1.0), exactly like collect_waypoint_demos.py. An episode ends 5 steps in a row
after FORGE says "inserted" (same rule as the scripted demos), or at the step limit. When it succeeded, press Enter:
the whole episode is sent to forge_server.py, which writes it ON THE CLUSTER, in --out-dir (relative to the project
root there, default demos/isaaclab/ForgePegInsertTeleop):

    ep_0000_seed100.npz   exterior_image, wrist_image, joint_position, gripper_position, ft_force,
                          peg_socket_offset, actions, prompt, seed  (same keys as the scripted demos)
    index.jsonl           one line per saved demo: seed, steps, wall seconds, final offset, max force, ...

then the next seed starts. Not saved: failed episodes, and successes you skip with R / Z (nothing is written
until you press Enter). The numbering ep_NNNN continues after the files already in the folder, so you can quit and
come back; use another --seed for the next session (the same seed gives the same start state).
The server must be restarted after pulling this version (new op "save_teleop_demo").

This file is self-contained on purpose: it imports nothing from the repo. Install (PowerShell, or WSL):

    py -m venv teleop-venv
    teleop-venv\\Scripts\\activate
    pip install numpy pygame websockets msgpack

Tunnel (second terminal, leave it open; <node> = the node of your salloc, e.g. cn-l042):

    ssh -J mila -L 8102:localhost:8102 <node>

Run:

    python teleop_client.py --host localhost --port 8102 --seed 100

Controls (Tab first: the mouse must be captured, otherwise nothing moves):
    mouse        move the peg in the horizontal plane, as seen in the WRIST camera (right image): mouse up = towards
                 the top of that image, mouse right = towards its right (the hand goes where you point; the scene
                 then slides the other way in the image). --view exterior uses the left image instead.
                 No button to hold: once captured, every movement counts. The pixels travelled since the previous
                 step are added up and sent as ONE move (a swipe = one step, capped at --max-xy-mm).
    arrow keys   the same directions: Up / Down / Left / Right = towards the top / bottom / left / right of the wrist
                 image. --key-xy-mm per step, repeated while the key is down; they add up with the mouse.
    W / S        peg up / down
    Shift        fine mode   (x 0.25: for the last millimetres)
    Ctrl         fast mode   (x 2.5)
    Space        one step with no move (let the arm settle)
    C            AUTO steps on / off (the simulation steps at every loop, also without input)
    Enter        after a SUCCESS: save the demo on the cluster and start the next seed
    Z            same seed again (discards the current episode: retry the same start state)
    R            next seed (discards the current episode)
    Tab          capture / release the mouse      Esc  quit
The caps per step (--max-xy-mm, --max-z-mm) are the same order as the scripted demos; the size of a move is
--xy-gain-mm per mouse pixel and --z-rate-mm per step with W / S. If a direction is wrong for your window, use
--flip-x / --flip-y / --rotate-deg (the directions are printed at every reset).
"""
import argparse
import collections
import dataclasses
import functools
import io
import statistics
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import websockets.exceptions
import websockets.sync.client


# ---- msgpack + numpy: same wire format as openpi_client.msgpack_numpy (copied, so no repo needed) ----
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---- Constants ------------------------------------------------------------------------------------
# Exterior camera pose of expo_ft/env/isaaclab/forge_env_patched.py (EXTERIOR_EYE / EXTERIOR_TARGET, env frame, m):
# only used to turn "mouse right / forward" into directions of the table plane as seen in the left image.
EXTERIOR_EYE = (1.05, 0.35, 0.35)
EXTERIOR_TARGET = (0.57, 0.02, 0.08)
# Wrist camera orientation in the fingertip frame (quaternion w x y z, ROS convention: image right = camera x, image
# down = camera y, optical axis = camera z). This is the Franka Hand camera; the server tells the one of its own robot
# (debug op "robot_info", see Session.start), this value is only the fallback for a server that does not know the op.
WRIST_ROT = (0.6916, -0.1471, -0.1471, 0.6916)
HOLD_AFTER_SUCCESS = 5      # same as the scripted demos: the episode ends this many steps after "inserted"
# observation key -> array name in the .npz (same as RECORDED of collect_waypoint_demos.py)
RECORDED = {
    "observation/exterior_image_1_left": "exterior_image",
    "observation/wrist_image_left": "wrist_image",
    "observation/joint_position": "joint_position",
    "observation/gripper_position": "gripper_position",
    "extra/ft_force": "ft_force",
    "extra/peg_socket_offset": "peg_socket_offset",
}
# saved only when the server sends it (robot with taxels): (T, 2, 7, 4) N, [:, 0] left finger, [:, 1] right
RECORDED_OPTIONAL = {"extra/tactile": "tactile", "extra/gripper_effort": "gripper_effort"}


# ---- Connection to forge_server.py ----------------------------------------------------------------
class RemoteEnv:
    """Same protocol as expo_ft/env/env_client.py (create_env / reset / debug), plus a timer on every call."""

    def __init__(self, host, port):
        self.uri = f"ws://{host}:{port}"
        self.packer = _Packer()
        self.timings = collections.defaultdict(lambda: collections.deque(maxlen=200))   # name -> ms
        self.env_id = None
        self._connect()

    def _connect(self):
        # ping_interval=None: no keepalive. Saving a demo sends ~10-30 MB through the tunnel in one message, which can
        # take longer than the default 20 s pong deadline and then closes the connection (seen on the cluster).
        kw = dict(compression=None, max_size=None, open_timeout=15, close_timeout=100)
        try:
            try:
                self.conn = websockets.sync.client.connect(self.uri, ping_interval=None, **kw)
            except TypeError:          # old websockets (< 15): no keepalive at all, nothing to disable
                self.conn = websockets.sync.client.connect(self.uri, **kw)
        except Exception as e:     # noqa: BLE001 -- any failure here means "no server behind the tunnel"
            raise SystemExit(
                f"cannot connect to {self.uri}: {type(e).__name__}: {e}\n"
                "  - is the ssh tunnel still open?   ssh -J mila -L <port>:localhost:<port> <node>\n"
                "  - is forge_server.py running on that node (and finished loading Isaac Sim)?") from e

    def reconnect(self):
        """New connection to the same server: the simulation (and the episode in progress) is untouched."""
        self.close()
        self._connect()
        self.create_env()

    def _call(self, name, operation, **request):
        t0 = time.perf_counter()
        self.conn.send(self.packer.pack({"operation": operation, **request}))
        response = _unpackb(self.conn.recv())
        self.timings[name].append((time.perf_counter() - t0) * 1000.0)
        if response.get("status", response.get("stats")) == "error":
            raise RuntimeError(f"server error in {operation}: {response.get('message')}")
        return response

    def create_env(self):
        response = self._call("create_env", "create_env")
        self.env_id = response["env_id"]
        return response["task_description"]

    def reset(self, seed):
        return self._call("reset", "reset", env_id=self.env_id, seed=int(seed))["observation"]

    def debug(self, kind, timer=None, **kw):
        result = self._call(timer or f"debug:{kind}", "debug", env_id=self.env_id, kind=kind, **kw)["result"]
        if isinstance(result, dict) and "error" in result and len(result) == 1:
            hint = "  (is forge_server.py an old version? pull and restart it)" if "unknown debug kind" in result["error"] else ""
            raise RuntimeError(result["error"] + hint)
        return result

    def teleop_step(self, delta_pos, target_quat, cmd_gain, max_joint_step):
        """One control step in ONE round trip: IK + step + observation + flags (forge_server._teleop_step)."""
        return self.debug("teleop_step", timer="teleop_step", delta_pos=[float(v) for v in delta_pos],
                          target_quat=[float(v) for v in target_quat], cmd_gain=float(cmd_gain),
                          max_joint_step=float(max_joint_step))

    def save_demo(self, out_dir, npz_bytes, seed, info, success):
        """The server writes the .npz (already built, compressed) in out_dir (relative to the project root there)."""
        return self.debug("save_teleop_demo", timer="save", out_dir=str(out_dir), npz=npz_bytes,
                          seed=int(seed), info=info, success=bool(success))

    def measure_rtt(self, n=20):
        """Round trip of a call that does nothing on the server: the network floor of every call."""
        for _ in range(n):
            self.debug("task_geometry", timer="rtt")

    def close(self):
        try:
            self.conn.close()
        except Exception:      # noqa: BLE001 -- already closed
            pass

    def mean_ms(self, name):
        values = self.timings.get(name)
        return statistics.fmean(values) if values else float("nan")

    def p95_ms(self, name):
        values = sorted(self.timings.get(name, ()))
        return values[int(0.95 * (len(values) - 1))] if values else float("nan")


# ---- Inputs -> move -------------------------------------------------------------------------------
@dataclasses.dataclass
class Inputs:
    """What the operator did since the previous step."""
    dx: int = 0              # mouse pixels, summed since the previous step (right +)
    dy: int = 0              # mouse pixels, summed since the previous step (down +)
    up: bool = False         # W
    down: bool = False       # S
    fine: bool = False       # Shift
    fast: bool = False       # Ctrl
    wait: bool = False       # Space: one step without moving
    left: bool = False       # arrow keys: horizontal moves in the directions of the exterior image
    right: bool = False
    forward: bool = False    # Up arrow: away from the camera
    back: bool = False       # Down arrow: towards the camera

    def any(self):
        return bool(self.dx or self.dy or self.up or self.down or self.wait
                    or self.left or self.right or self.forward or self.back)


def screen_axes(flip_x=False, flip_y=False, rotate_deg=0.0):
    """Unit vectors (x, y of the env frame) of 'mouse right' and 'mouse forward' for the exterior camera.

    forward = the horizontal direction the camera looks along (away from the camera); right = to the right of the
    image (forward x up). The robot base is at the env origin; the camera looks at the table from the +x side.
    """
    f = np.asarray(EXTERIOR_TARGET[:2], dtype=np.float64) - np.asarray(EXTERIOR_EYE[:2], dtype=np.float64)
    forward = f / np.linalg.norm(f)
    right = np.array([forward[1], -forward[0]])
    a = np.deg2rad(rotate_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    right, forward = rot @ right, rot @ forward
    if flip_x:
        right = -right
    if flip_y:
        forward = -forward
    return right, forward


def quat_to_matrix(q):
    """3x3 rotation matrix of a quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrist_axes(hand_quat, flip_x=False, flip_y=False, rotate_deg=0.0, wrist_rot=WRIST_ROT):
    """Unit vectors (x, y of the env frame) of 'towards the right' and 'towards the top' of the WRIST image.

    The camera is fixed on the hand, whose orientation is held (hand_quat = fingertip quaternion of the server,
    w x y z): image right / up are the camera's x / -y axes in the env frame, projected on the horizontal plane.
    The hand looks down, so the camera is nearly vertical (tilted ~24 deg) and these projections are well defined.
    """
    rot = quat_to_matrix(hand_quat) @ quat_to_matrix(wrist_rot)
    right, up = rot[:2, 0], -rot[:2, 1]
    right, up = right / np.linalg.norm(right), up / np.linalg.norm(up)
    a = np.deg2rad(rotate_deg)
    r2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    right, up = r2 @ right, r2 @ up
    return (-right if flip_x else right), (-up if flip_y else up)


def view_axes(args, hand_quat, wrist_rot=WRIST_ROT):
    """Directions of mouse right / mouse up in the env frame, for the chosen view (wrist or exterior image)."""
    if args.view == "exterior":
        return screen_axes(args.flip_x, args.flip_y, args.rotate_deg)
    return wrist_axes(hand_quat, args.flip_x, args.flip_y, args.rotate_deg, wrist_rot)


def build_delta(inp, axes, xy_gain_mm, z_rate_mm, max_xy_mm, max_z_mm, key_xy_mm=2.0):
    """The peg translation (m, env frame) of one control step, from the inputs. Capped per step."""
    scale = 0.25 if inp.fine else (2.5 if inp.fast else 1.0)
    right, forward = axes
    # mouse (pygame's y grows downwards) + arrow keys, both in the directions of the exterior image
    xy = xy_gain_mm * scale * (inp.dx * right + (-inp.dy) * forward)
    xy = xy + key_xy_mm * scale * ((int(inp.right) - int(inp.left)) * right + (int(inp.forward) - int(inp.back)) * forward)
    norm = float(np.linalg.norm(xy))
    if norm > max_xy_mm:
        xy = xy * (max_xy_mm / norm)
    z = float(np.clip((int(inp.up) - int(inp.down)) * z_rate_mm * scale, -max_z_mm, max_z_mm))
    return np.array([xy[0], xy[1], z]) * 1e-3


# ---- One teleoperated episode ---------------------------------------------------------------------
class Session:
    """State of the current episode. No pygame in here (so it can be tested without a screen)."""

    def __init__(self, env, args):
        self.env, self.args = env, args
        self.seed = None
        self.obs = None
        self.quat0 = None
        self.prompt = ""

    def start(self, seed):
        self.seed = seed
        self.obs = self.env.reset(seed)
        self.quat0 = self.env.debug("fingertip_pose")["quat"]       # orientation held for the whole episode
        if getattr(self, "wrist_rot", None) is None:                # camera of the server's robot (asked once)
            robot = self.env.debug("robot_info")
            rot = robot.get("wrist_rot_ros_in_fingertip") if isinstance(robot, dict) else None
            self.wrist_rot = tuple(rot) if rot else WRIST_ROT
            self.robot_name = robot.get("name", "?") if isinstance(robot, dict) else "?"
        self.axes = view_axes(self.args, self.quat0, self.wrist_rot)   # mouse right / up in the env frame
        self.step_count = 0
        self.success_run = 0
        self.success = False
        self.timed_out = False
        self.over = False
        self.clipped_steps = 0
        self.last_delta = np.zeros(3)
        self.last_clipped = False
        self.recorded = {**RECORDED, **{k: v for k, v in RECORDED_OPTIONAL.items() if k in self.obs}}
        self.frames = {name: [] for name in self.recorded.values()}   # what is saved: the observations ...
        self.actions = []                                           # ... and the actions sent
        self.idle_steps = 0                                         # steps without any move
        self.max_force = 0.0
        self.t_first = None                                         # wall clock of the first / last step
        self.t_last = None
        self.saved = False

    def step(self, delta):
        if self.t_first is None:
            self.t_first = time.perf_counter()
        # the observation the action is computed from (the one on screen), like the scripted demos
        for key, name in self.recorded.items():
            self.frames[name].append(np.asarray(self.obs[key]))
        r = self.env.teleop_step(delta, self.quat0, self.args.cmd_gain, self.args.max_joint_step)
        self.t_last = time.perf_counter()
        self.actions.append(np.asarray(r["action"], dtype=np.float32))
        self.idle_steps += int(not np.any(delta))
        self.max_force = max(self.max_force, float(np.linalg.norm(np.asarray(self.obs["extra/ft_force"], dtype=np.float64))))
        self.last_delta, self.last_clipped = np.asarray(delta, dtype=np.float64), bool(r["clipped"])
        self.clipped_steps += int(self.last_clipped)
        self.step_count += 1
        if r["done"]:
            # the env reset itself (timeout / termination): its observation is the first one of a NEW
            # episode, so keep showing the last observation of this one.
            self.timed_out = True
            self.over = True
            return
        self.obs = r["obs"]
        self.success_run = self.success_run + 1 if r["success"] else 0
        if self.success_run >= HOLD_AFTER_SUCCESS:
            self.success, self.over = True, True
        if self.step_count >= self.args.max_steps:
            self.over = True

    def arrays(self):
        out = {name: np.stack(values) for name, values in self.frames.items()}
        out["actions"] = np.stack(self.actions)
        return out

    def info(self):
        """One line of index.jsonl (plain python types: it is sent to the server and written as JSON)."""
        final = np.asarray(self.obs["extra/peg_socket_offset"], dtype=np.float64) * 1000.0
        return {"seed": int(self.seed), "success": bool(self.success), "teleop": True, "steps": len(self.actions),
                "idle_steps": int(self.idle_steps), "clipped_steps": int(self.clipped_steps),
                "wall_seconds": round(float(self.t_last - self.t_first), 1) if self.t_first else 0.0,
                "final_offset_mm": [round(float(v), 2) for v in final], "max_force_N": round(self.max_force, 2)}

    def save(self):
        """Send the finished episode to the server, which writes it on the cluster. True when it worked."""
        if self.saved or not self.success or not self.args.out_dir:
            return False
        # the .npz is built here (losslessly compressed: 3-5x smaller to send) and written as is by the server
        t0 = time.perf_counter()
        buf = io.BytesIO()
        np.savez_compressed(buf, **self.arrays(), prompt=np.array(self.prompt), seed=np.array(self.seed))
        data = buf.getvalue()
        print(f"sending the demo ({len(self.actions)} steps, {len(data) / 1e6:.1f} MB) to the cluster ...", flush=True)
        r = None
        for attempt in (1, 2):
            try:
                r = self.env.save_demo(self.args.out_dir, data, self.seed, self.info(), True)
                break
            except websockets.exceptions.ConnectionClosed as e:     # lost the tunnel / server: reconnect, once
                print(f"   connection lost ({type(e).__name__}), reconnecting ...", flush=True)
                try:
                    self.env.reconnect()
                except Exception as e2:     # noqa: BLE001
                    print(f"   cannot reconnect: {e2}")
                    break
            except Exception as e:     # noqa: BLE001 -- refused by the server (bad out_dir, ...), nothing to retry
                print(f"!! NOT saved: {e}")
                break
        if r is None:
            # never lose a demo: keep a copy next to the client (copy it to the cluster by hand, e.g. with scp)
            backup = Path(f"teleop_unsaved_seed{self.seed}_{time.strftime('%H%M%S')}.npz")
            backup.write_bytes(data)
            print(f"!! NOT saved on the cluster -- a copy is in {backup.resolve()} ; Enter tries again")
            return False
        print(f"   ({time.perf_counter() - t0:.1f} s)")
        self.saved = True
        print(f"saved {r['file']}  ({len(self.actions)} steps)  ->  {r['path']}   [{r['count']} demos in the folder]")
        return True

    def offset_mm(self):
        return np.asarray(self.obs["extra/peg_socket_offset"], dtype=np.float64) * 1000.0

    def force_n(self):
        return float(np.linalg.norm(np.asarray(self.obs["extra/ft_force"], dtype=np.float64)))


# ---- Display --------------------------------------------------------------------------------------
BACKGROUND = (18, 18, 22)
TEXT = (230, 230, 230)
DIM = (140, 140, 150)
GOOD = (90, 200, 120)
WARN = (240, 190, 70)
BAD = (235, 90, 90)


def to_surface(pygame, image, scale):
    """HxWx3 uint8 array -> pygame surface, enlarged by an integer factor (nearest neighbour)."""
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8).swapaxes(0, 1))   # pygame wants (W, H, 3)
    surface = pygame.surfarray.make_surface(array)
    return pygame.transform.scale(surface, (array.shape[0] * scale, array.shape[1] * scale))


def draw_text(screen, font, x, y, text, colour=TEXT):
    screen.blit(font.render(text, True, colour), (x, y))
    return y + font.get_linesize()


def main():
    p = argparse.ArgumentParser(description="Teleop client, step 3: drive the arm with the mouse and the keyboard, "
                                            "record demos on the cluster.")
    p.add_argument("--host", default="localhost", help="Local end of the ssh tunnel.")
    p.add_argument("--port", type=int, default=8102, help="Local port of the tunnel (server_port of the YAML).")
    p.add_argument("--seed", type=int, default=100,
                   help="Reset seed of the first episode; the next episode (R, or after a save) uses seed + 1, ... "
                        "Use another value for each session: the same seed gives the same start state.")
    p.add_argument("--out-dir", default="demos/isaaclab/ForgePegInsertTeleop",
                   help="Folder ON THE CLUSTER where the server writes the demos, relative to the project root there "
                        "(the folder that holds expo_ft/). Must stay inside the project. '' = record nothing.")
    p.add_argument("--soft-limit-steps", type=int, default=90,
                   help="The step counter turns orange beyond this many steps (the policy has 150; shorter demos "
                        "are closer to the scripted ones). Display only.")
    p.add_argument("--scale", type=int, default=3, help="Camera enlargement factor (224 px x scale).")
    p.add_argument("--max-steps", type=int, default=150, help="Episode length (YAML max_steps_per_episode).")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="Quit by itself after this time and print the latency summary (0 = no limit).")
    p.add_argument("--continuous", action="store_true",
                   help="Start in AUTO mode: step the simulation at every loop (--hz), also without input, like a "
                        "real-time teleop. Default: step only when you give an input (slow motion). "
                        "The C key switches between the two modes during the session.")
    p.add_argument("--hz", type=float, default=15.0, help="Loop rate of --continuous (the server may be slower).")
    p.add_argument("--hold", action="store_true",
                   help="Do not move the arm (step 1 behaviour): continuous zero moves, to measure the latency.")
    # size of the moves
    p.add_argument("--xy-gain-mm", type=float, default=0.1, help="Peg move (mm) per mouse pixel, in the horizontal plane.")
    p.add_argument("--z-rate-mm", type=float, default=4.0, help="Peg move (mm) per step while W / S is down.")
    p.add_argument("--key-xy-mm", type=float, default=2.0,
                   help="Peg move (mm) per step while an arrow key is down (Shift x0.25, Ctrl x2.5).")
    p.add_argument("--max-xy-mm", type=float, default=10.0, help="Largest horizontal move per step (mm).")
    p.add_argument("--max-z-mm", type=float, default=15.0, help="Largest vertical move per step (mm).")
    p.add_argument("--cmd-gain", type=float, default=1.0,
                   help="Multiplies every joint-target change (as --cmd-gain of collect_waypoint_demos.py).")
    p.add_argument("--max-joint-step", type=float, default=0.05, help="Safety: largest joint-target change (rad).")
    # mouse and arrow directions
    p.add_argument("--view", choices=("wrist", "exterior"), default="wrist",
                   help="Image the mouse and the arrows refer to: the wrist camera (right image, default) or the "
                        "exterior camera (left image).")
    p.add_argument("--flip-x", action="store_true", help="Mouse right moves the peg to the left of the image.")
    p.add_argument("--flip-y", action="store_true", help="Mouse forward moves the peg towards the camera.")
    p.add_argument("--rotate-deg", type=float, default=0.0, help="Rotate the mouse directions in the table plane.")
    args = p.parse_args()

    import pygame
    pygame.init()
    # keys whose short taps (between two slow steps) must not be missed: remembered until the next step
    LATCHED_KEYS = (pygame.K_w, pygame.K_s, pygame.K_SPACE, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN)
    size = 224 * args.scale
    hud_height = 284
    screen = pygame.display.set_mode((max(2 * size, 1000), size + hud_height))
    pygame.display.set_caption("FORGE PegInsert -- teleop client (step 3)")
    font = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 18)
    big_font = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 28, bold=True)
    clock = pygame.time.Clock()

    env = RemoteEnv(args.host, args.port)
    prompt = env.create_env()
    geom = env.debug("task_geometry")
    print(f"connected to {env.uri}   task: {prompt!r}")
    print(f"hole depth {geom['hole_depth'] * 1000:.0f} mm   success: xy < {geom['success_xy'] * 1000:.1f} mm, "
          f"z < {geom['success_z'] * 1000:.1f} mm")
    env.measure_rtt()
    print(f"network round trip (empty call): {env.mean_ms('rtt'):.1f} ms")

    session = Session(env, args)
    session.prompt = prompt
    session.start(args.seed)
    print(f"recording: demos are written on the cluster in <project>/{args.out_dir}" if args.out_dir
          else "recording OFF (--out-dir '')")
    saved_count = 0                                  # demos saved by this run

    def describe_axes():
        right, up = session.axes
        return (f"{args.view} image: right -> env (x, y) = ({right[0]:+.2f}, {right[1]:+.2f})   "
                f"up -> ({up[0]:+.2f}, {up[1]:+.2f})")
    print(describe_axes())
    print(f"observation keys: {sorted(session.obs.keys())}")
    print("episode", session.seed, "started.  Tab = capture the mouse, R = new episode, Esc = quit.")

    grabbed = False
    auto = bool(args.continuous)              # C toggles: True = the simulation steps at every loop, also without input
    acc = np.zeros(2, dtype=np.int64)         # mouse motion since the previous step
    latched = set()                           # W / S / Space pressed since the previous step (short taps)
    shown = Inputs()                          # last inputs, for the HUD
    t_start = time.perf_counter()
    last_print = t_start
    last_logged = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_r, pygame.K_z, pygame.K_RETURN, pygame.K_KP_ENTER):
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if not session.success:
                            continue                                  # nothing to save yet
                        if not session.saved and not session.save():
                            continue                                  # the error is printed, Enter tries again
                        saved_count += int(session.saved)
                        new_seed = session.seed + 1
                    else:
                        if session.success and not session.saved:
                            print(f"episode {session.seed}: successful but NOT saved (Enter saves it) -- discarded")
                        new_seed = session.seed if event.key == pygame.K_z else session.seed + 1
                    session.start(new_seed)
                    auto = bool(args.continuous)
                    acc[:] = 0
                    latched.clear()
                    print("episode", session.seed, "started" + (" again (same start state)" if event.key == pygame.K_z else "")
                          + ".   " + describe_axes())
                elif event.key == pygame.K_TAB:
                    grabbed = not grabbed
                    pygame.event.set_grab(grabbed)
                    pygame.mouse.set_visible(not grabbed)
                    pygame.mouse.get_rel()                   # forget the jump caused by the capture
                    acc[:] = 0
                elif event.key == pygame.K_c:
                    auto = not auto
                    print(f"step {session.step_count}: {'AUTO mode (steps counted continuously)' if auto else 'slow motion (steps only on input)'}")
                elif event.key in LATCHED_KEYS:
                    latched.add(event.key)
        keys = pygame.key.get_pressed()
        dx, dy = pygame.mouse.get_rel()
        if grabbed:
            acc += (dx, dy)
        else:
            acc[:] = 0
            latched.clear()

        inp = Inputs(dx=int(acc[0]), dy=int(acc[1]),
                     up=bool(keys[pygame.K_w] or pygame.K_w in latched),
                     down=bool(keys[pygame.K_s] or pygame.K_s in latched),
                     fine=bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]),
                     fast=bool(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]),
                     wait=pygame.K_SPACE in latched,
                     left=bool(keys[pygame.K_LEFT] or pygame.K_LEFT in latched),
                     right=bool(keys[pygame.K_RIGHT] or pygame.K_RIGHT in latched),
                     forward=bool(keys[pygame.K_UP] or pygame.K_UP in latched),
                     back=bool(keys[pygame.K_DOWN] or pygame.K_DOWN in latched))
        stepping = False
        if not session.over:
            if args.hold or auto:
                stepping = grabbed or args.hold
            else:
                stepping = grabbed and inp.any()
        if stepping:
            if args.hold:
                inp = Inputs()
            delta = build_delta(inp, session.axes, args.xy_gain_mm, args.z_rate_mm, args.max_xy_mm, args.max_z_mm,
                                args.key_xy_mm)
            session.step(delta)
            shown = inp
            acc[:] = 0
            latched.clear()
            if session.success:
                print(f"episode {session.seed}: SUCCESS after {session.step_count} steps")
            elif session.timed_out:
                print(f"episode {session.seed}: over after {session.step_count} steps (no success)")

        # -- draw --
        screen.fill(BACKGROUND)
        screen.blit(to_surface(pygame, session.obs["observation/exterior_image_1_left"], args.scale), (0, 0))
        screen.blit(to_surface(pygame, session.obs["observation/wrist_image_left"], args.scale), (size, 0))
        offset_mm = session.offset_mm()
        xy = float(np.hypot(offset_mm[0], offset_mm[1]))
        xy_colour = GOOD if xy < geom["success_xy"] * 1000 else TEXT
        y = size + 6
        x = 10
        for text, colour in ((f"xy {xy:6.2f} mm", xy_colour), (f"z {offset_mm[2]:5.1f} mm", TEXT),
                             (f"x {offset_mm[0]:+6.2f}  y {offset_mm[1]:+6.2f}", DIM)):
            rendered = big_font.render(text, True, colour)
            screen.blit(rendered, (x, y))
            x += rendered.get_width() + 30
        y += 42
        if session.success and session.saved:
            status, status_colour = "SAVED", GOOD
        elif session.success:
            status, status_colour = ("SUCCESS -- Enter = SAVE + next seed, Z = redo, R = skip" if args.out_dir
                                     else "SUCCESS (recording off) -- R = next seed"), GOOD
        elif session.over:
            status, status_colour = "episode over -- Z = same seed again, R = next seed", WARN
        elif not grabbed and not args.hold:
            status, status_colour = "mouse FREE -- press Tab to take control", WARN
        else:
            status, status_colour = ("AUTO: steps counted continuously (C = stop)" if auto or args.hold else "your move (C = auto steps)"), (WARN if auto else TEXT)
        over_soft = session.step_count > args.soft_limit_steps and not session.over
        y = draw_text(screen, font, 10, y,
                      f"seed {session.seed}   step {session.step_count}/{args.max_steps}   {status}",
                      WARN if over_soft and status_colour == TEXT else status_colour)
        d = session.last_delta * 1000.0 + 0.0            # "+ 0.0": no "-0.00" for a negative zero
        force = session.force_n()
        y = draw_text(screen, font, 10, y,
                      f"last move (mm)  x {d[0]:+6.2f}  y {d[1]:+6.2f}  z {d[2]:+6.2f}"
                      f"{'   JOINT CLIP' if session.last_clipped else ''}"
                      f"      contact force {force:5.2f} N", WARN if force > 2.0 else TEXT)
        total = env.mean_ms("teleop_step")
        y = draw_text(screen, font, 10, y,
                      f"saved this run: {saved_count}   (success hold {session.success_run}/{HOLD_AFTER_SUCCESS})"
                      + ("   -- long demo, try to be quicker" if over_soft else ""), WARN if over_soft else DIM)
        y = draw_text(screen, font, 10, y,
                      f"ms mean/p95:  teleop_step {total:.0f}/{env.p95_ms('teleop_step'):.0f}   "
                      f"empty call {env.mean_ms('rtt'):.0f}   -> {1000.0 / total if total == total else float('nan'):.1f} steps/s max",
                      DIM)
        mode = "FINE" if shown.fine else ("FAST" if shown.fast else "normal")
        down_keys = [name for name, key in (("W", pygame.K_w), ("S", pygame.K_s), ("Space", pygame.K_SPACE),
                                            ("Left", pygame.K_LEFT), ("Right", pygame.K_RIGHT), ("Up", pygame.K_UP),
                                            ("Down", pygame.K_DOWN), ("Shift", pygame.K_LSHIFT),
                                            ("Ctrl", pygame.K_LCTRL)) if keys[key]]
        y = draw_text(screen, font, 10, y,
                      f"inputs: mouse pending ({acc[0]:+4d},{acc[1]:+4d}) px   keys: {' '.join(down_keys) or '-'}   "
                      f"mode: {mode}   mouse {'CAPTURED' if grabbed else 'free'}", DIM)
        y = draw_text(screen, font, 10, y,
                      f"{args.view} image: right -> env ({session.axes[0][0]:+.2f},{session.axes[0][1]:+.2f})  up -> "
                      f"({session.axes[1][0]:+.2f},{session.axes[1][1]:+.2f})   mouse {args.xy_gain_mm:g} mm/px   "
                      f"arrows {args.key_xy_mm:g} mm/step   "
                      f"W/S {args.z_rate_mm:g} mm/step", DIM)
        y = draw_text(screen, font, 10, y,
                      "Tab capture  mouse/arrows = xy (wrist)  W/S up/down  Shift fine  Ctrl fast  Space wait", DIM)
        draw_text(screen, font, 10, y,
                  "C auto steps  Enter save (after SUCCESS)  Z same seed again  R next seed  Esc quit", DIM)
        pygame.display.flip()

        # -- pace the loop --
        if auto or args.hold:
            clock.tick(args.hz if args.hz > 0 else 0)
        else:
            clock.tick(60)                      # idle: only redraw, the simulation is not stepped
        now = time.perf_counter()
        if now - last_print > 5.0 and env.timings.get("teleop_step"):
            last_print = now
            if (session.seed, session.step_count) != last_logged:      # nothing new while idle / after the end
                last_logged = (session.seed, session.step_count)
                print(f"  teleop_step {env.mean_ms('teleop_step'):.0f} ms (p95 {env.p95_ms('teleop_step'):.0f})   "
                      f"step {session.step_count}")
        if args.max_seconds and now - t_start > args.max_seconds:
            running = False

    print("\nround trips (ms)        mean     p95")
    for name in ("rtt", "teleop_step"):
        if env.timings.get(name):
            label = "empty call (network)" if name == "rtt" else "teleop_step"
            print(f"  {label:<20s} {env.mean_ms(name):7.1f} {env.p95_ms(name):7.1f}")
    if env.timings.get("teleop_step"):
        total = env.mean_ms("teleop_step")
        print(f"  -> {1000.0 / total:.1f} steps/s at most; a {args.max_steps}-step episode needs at least "
              f"{args.max_steps * total / 1000.0:.0f} s of pure simulation")
    env.close()
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
