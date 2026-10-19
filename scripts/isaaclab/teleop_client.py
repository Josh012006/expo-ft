"""Teleoperation client for the FORGE PegInsert simulation -- STEP 2/3: you drive the arm (nothing is recorded yet).

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
is recorded in step 3 is therefore at the policy's 15 Hz, in the same format as the scripted demos.

This file is self-contained on purpose: it imports nothing from the repo. Install (PowerShell, or WSL):

    py -m venv teleop-venv
    teleop-venv\\Scripts\\activate
    pip install numpy pygame websockets msgpack

Tunnel (second terminal, leave it open; <node> = the node of your salloc, e.g. cn-l042):

    ssh -J mila -L 8102:localhost:8102 <node>

Run:

    python teleop_client.py --host localhost --port 8102 --seed 100

Controls (Tab first: the mouse must be captured, otherwise nothing moves):
    mouse        move the peg in the horizontal plane, in the directions of the EXTERIOR camera image:
                 mouse right = to the right of the left image, mouse forward (up) = away from the camera
    W / S        peg up / down
    Shift        fine mode   (x 0.25: for the last millimetres)
    Ctrl         fast mode   (x 2.5)
    Space        one step with no move (let the arm settle)
    R            new episode (next seed)      Tab  capture / release the mouse      Esc  quit
The caps per step (--max-xy-mm, --max-z-mm) are the same order as the scripted demos; the size of a move is
--xy-gain-mm per mouse pixel and --z-rate-mm per step with W / S. If a direction is wrong for your window, use
--flip-x / --flip-y / --rotate-deg (printed at start-up).
"""
import argparse
import collections
import dataclasses
import functools
import statistics
import sys
import time

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
HOLD_AFTER_SUCCESS = 5      # same as the scripted demos: the episode ends this many steps after "inserted"


# ---- Connection to forge_server.py ----------------------------------------------------------------
class RemoteEnv:
    """Same protocol as expo_ft/env/env_client.py (create_env / reset / debug), plus a timer on every call."""

    def __init__(self, host, port):
        self.uri = f"ws://{host}:{port}"
        try:
            self.conn = websockets.sync.client.connect(
                self.uri, compression=None, max_size=None, open_timeout=15, close_timeout=100)
        except Exception as e:     # noqa: BLE001 -- any failure here means "no server behind the tunnel"
            raise SystemExit(
                f"cannot connect to {self.uri}: {type(e).__name__}: {e}\n"
                "  - is the ssh tunnel still open?   ssh -J mila -L <port>:localhost:<port> <node>\n"
                "  - is forge_server.py running on that node (and finished loading Isaac Sim)?") from e
        self.packer = _Packer()
        self.timings = collections.defaultdict(lambda: collections.deque(maxlen=200))   # name -> ms
        self.env_id = None

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
            raise RuntimeError(f"{result['error']}  (is forge_server.py an old version? pull and restart it)")
        return result

    def teleop_step(self, delta_pos, target_quat, cmd_gain, max_joint_step):
        """One control step in ONE round trip: IK + step + observation + flags (forge_server._teleop_step)."""
        return self.debug("teleop_step", timer="teleop_step", delta_pos=[float(v) for v in delta_pos],
                          target_quat=[float(v) for v in target_quat], cmd_gain=float(cmd_gain),
                          max_joint_step=float(max_joint_step))

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

    def any(self):
        return bool(self.dx or self.dy or self.up or self.down or self.wait)


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


def build_delta(inp, axes, xy_gain_mm, z_rate_mm, max_xy_mm, max_z_mm):
    """The peg translation (m, env frame) of one control step, from the inputs. Capped per step."""
    scale = 0.25 if inp.fine else (2.5 if inp.fast else 1.0)
    right, forward = axes
    xy = xy_gain_mm * scale * (inp.dx * right + (-inp.dy) * forward)        # mm; pygame's y grows downwards
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

    def start(self, seed):
        self.seed = seed
        self.obs = self.env.reset(seed)
        self.quat0 = self.env.debug("fingertip_pose")["quat"]       # orientation held for the whole episode
        self.step_count = 0
        self.success_run = 0
        self.success = False
        self.timed_out = False
        self.over = False
        self.clipped_steps = 0
        self.last_delta = np.zeros(3)
        self.last_clipped = False

    def step(self, delta):
        r = self.env.teleop_step(delta, self.quat0, self.args.cmd_gain, self.args.max_joint_step)
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
    p = argparse.ArgumentParser(description="Teleop client, step 2: drive the arm with the mouse and the keyboard.")
    p.add_argument("--host", default="localhost", help="Local end of the ssh tunnel.")
    p.add_argument("--port", type=int, default=8102, help="Local port of the tunnel (server_port of the YAML).")
    p.add_argument("--seed", type=int, default=100, help="Reset seed of the first episode; R uses seed + 1, ...")
    p.add_argument("--scale", type=int, default=3, help="Camera enlargement factor (224 px x scale).")
    p.add_argument("--max-steps", type=int, default=150, help="Episode length (YAML max_steps_per_episode).")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="Quit by itself after this time and print the latency summary (0 = no limit).")
    p.add_argument("--continuous", action="store_true",
                   help="Step the simulation at every loop (--hz), also without input, like a real-time teleop. "
                        "Default: step only when you give an input (slow motion).")
    p.add_argument("--hz", type=float, default=15.0, help="Loop rate of --continuous (the server may be slower).")
    p.add_argument("--hold", action="store_true",
                   help="Do not move the arm (step 1 behaviour): continuous zero moves, to measure the latency.")
    # size of the moves
    p.add_argument("--xy-gain-mm", type=float, default=0.1, help="Peg move (mm) per mouse pixel, in the horizontal plane.")
    p.add_argument("--z-rate-mm", type=float, default=4.0, help="Peg move (mm) per step while W / S is down.")
    p.add_argument("--max-xy-mm", type=float, default=10.0, help="Largest horizontal move per step (mm).")
    p.add_argument("--max-z-mm", type=float, default=15.0, help="Largest vertical move per step (mm).")
    p.add_argument("--cmd-gain", type=float, default=1.0,
                   help="Multiplies every joint-target change (as --cmd-gain of collect_waypoint_demos.py).")
    p.add_argument("--max-joint-step", type=float, default=0.05, help="Safety: largest joint-target change (rad).")
    # mouse directions
    p.add_argument("--flip-x", action="store_true", help="Mouse right moves the peg to the left of the image.")
    p.add_argument("--flip-y", action="store_true", help="Mouse forward moves the peg towards the camera.")
    p.add_argument("--rotate-deg", type=float, default=0.0, help="Rotate the mouse directions in the table plane.")
    args = p.parse_args()

    import pygame
    pygame.init()
    size = 224 * args.scale
    hud_height = 232
    screen = pygame.display.set_mode((max(2 * size, 1000), size + hud_height))
    pygame.display.set_caption("FORGE PegInsert -- teleop client (step 2)")
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

    axes = screen_axes(args.flip_x, args.flip_y, args.rotate_deg)
    print(f"mouse right   -> env (x, y) = ({axes[0][0]:+.2f}, {axes[0][1]:+.2f})   "
          f"mouse forward -> ({axes[1][0]:+.2f}, {axes[1][1]:+.2f})")
    session = Session(env, args)
    session.start(args.seed)
    print(f"observation keys: {sorted(session.obs.keys())}")
    print("episode", session.seed, "started.  Tab = capture the mouse, R = new episode, Esc = quit.")

    grabbed = False
    acc = np.zeros(2, dtype=np.int64)         # mouse motion since the previous step
    latched = set()                           # W / S / Space pressed since the previous step (short taps)
    shown = Inputs()                          # last inputs, for the HUD
    t_start = time.perf_counter()
    last_print = t_start
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    session.start(session.seed + 1)
                    acc[:] = 0
                    latched.clear()
                    print("episode", session.seed, "started.")
                elif event.key == pygame.K_TAB:
                    grabbed = not grabbed
                    pygame.event.set_grab(grabbed)
                    pygame.mouse.set_visible(not grabbed)
                    pygame.mouse.get_rel()                   # forget the jump caused by the capture
                    acc[:] = 0
                elif event.key in (pygame.K_w, pygame.K_s, pygame.K_SPACE):
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
                     wait=pygame.K_SPACE in latched)
        stepping = False
        if not session.over:
            if args.hold or args.continuous:
                stepping = grabbed or args.hold
            else:
                stepping = grabbed and inp.any()
        if stepping:
            if args.hold:
                inp = Inputs()
            delta = build_delta(inp, axes, args.xy_gain_mm, args.z_rate_mm, args.max_xy_mm, args.max_z_mm)
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
        if session.success:
            status, status_colour = "SUCCESS -- press R", GOOD
        elif session.over:
            status, status_colour = "episode over -- press R", WARN
        elif not grabbed and not args.hold:
            status, status_colour = "mouse FREE -- press Tab to take control", WARN
        else:
            status, status_colour = ("running" if args.continuous or args.hold else "your move"), TEXT
        y = draw_text(screen, font, 10, y,
                      f"seed {session.seed}   step {session.step_count}/{args.max_steps}   {status}", status_colour)
        d = session.last_delta * 1000.0 + 0.0            # "+ 0.0": no "-0.00" for a negative zero
        force = session.force_n()
        y = draw_text(screen, font, 10, y,
                      f"last move (mm)  x {d[0]:+6.2f}  y {d[1]:+6.2f}  z {d[2]:+6.2f}"
                      f"{'   JOINT CLIP' if session.last_clipped else ''}"
                      f"      contact force {force:5.2f} N", WARN if force > 2.0 else TEXT)
        total = env.mean_ms("teleop_step")
        y = draw_text(screen, font, 10, y,
                      f"ms mean/p95:  teleop_step {total:.0f}/{env.p95_ms('teleop_step'):.0f}   "
                      f"empty call {env.mean_ms('rtt'):.0f}   -> {1000.0 / total if total == total else float('nan'):.1f} steps/s max",
                      DIM)
        mode = "FINE" if shown.fine else ("FAST" if shown.fast else "normal")
        down_keys = [name for name, key in (("W", pygame.K_w), ("S", pygame.K_s), ("Space", pygame.K_SPACE),
                                            ("Shift", pygame.K_LSHIFT), ("Ctrl", pygame.K_LCTRL)) if keys[key]]
        y = draw_text(screen, font, 10, y,
                      f"inputs: mouse pending ({acc[0]:+4d},{acc[1]:+4d}) px   keys: {' '.join(down_keys) or '-'}   "
                      f"mode: {mode}   mouse {'CAPTURED' if grabbed else 'free'}", DIM)
        y = draw_text(screen, font, 10, y,
                      f"mouse right -> env ({axes[0][0]:+.2f},{axes[0][1]:+.2f})  forward -> ({axes[1][0]:+.2f},"
                      f"{axes[1][1]:+.2f})   gain {args.xy_gain_mm:g} mm/px   W/S {args.z_rate_mm:g} mm/step", DIM)
        draw_text(screen, font, 10, y,
                  "Tab capture   mouse xy   W/S up/down   Shift fine   Ctrl fast   Space wait   R new episode   Esc quit", DIM)
        pygame.display.flip()

        # -- pace the loop --
        if args.continuous or args.hold:
            clock.tick(args.hz if args.hz > 0 else 0)
        else:
            clock.tick(60)                      # idle: only redraw, the simulation is not stepped
        now = time.perf_counter()
        if now - last_print > 5.0 and env.timings.get("teleop_step"):
            last_print = now
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
