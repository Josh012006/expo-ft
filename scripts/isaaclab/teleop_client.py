"""Teleoperation client for the FORGE PegInsert simulation -- STEP 1/3: viewer + input test + latency.

Runs on YOUR PC (Windows is fine), not on the cluster. The simulation keeps running in forge_server.py
on an L40S node; this script connects to it through an SSH tunnel and shows what the policy will see:

    left   exterior camera (224x224, enlarged)
    right  wrist camera    (224x224, enlarged)
    under  the offset peg -> hole (ground truth, simulation only), the state of the keyboard / mouse and
           the time of every round trip to the server

It does NOT move the arm yet (step 2) and records nothing (step 3). Each loop sends a "hold" action
(= the measured joint angles, gripper closed) so the arm stays where it is. The loop does exactly the
server calls that step 2 will do (IK + step + observation + info), so the numbers you read here are
the real cost of teleoperating.

This file is self-contained on purpose: it imports nothing from the repo, so it runs in a tiny
Windows environment.  Install (PowerShell):

    py -m venv teleop-venv
    teleop-venv\\Scripts\\activate
    pip install numpy pygame websockets msgpack

Tunnel (second PowerShell, leave it open; <node> = the node of your salloc, e.g. cn-l042):

    ssh -J mila -L 8102:localhost:8102 <node>

Run:

    python scripts\\isaaclab\\teleop_client.py --host localhost --port 8102 --seed 100

Keys:  R = new episode (next seed)    Tab = capture / release the mouse    Esc = quit
       The mouse movement and the keys W S Shift Ctrl are only displayed (they drive the arm in step 2).
"""
import argparse
import collections
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


# ---- Connection to forge_server.py ---------------------------------------------------------------
GRIPPER_CLOSED = 1.0


class RemoteEnv:
    """Same protocol as expo_ft/env/env_client.py (create_env / reset / step / get_observation /
    get_info_for_step / debug), plus a timer on every call."""

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

    def step(self, action):
        self._call("step", "step", env_id=self.env_id, action=np.asarray(action, dtype=np.float32))

    def get_observation(self):
        return self._call("obs", "get_observation", env_id=self.env_id)["observation"]

    def get_info(self):
        r = self._call("info", "get_info_for_step", env_id=self.env_id)
        return bool(r["done"]), bool(r["success"])

    def debug(self, kind, timer=None, **kw):
        return self._call(timer or f"debug:{kind}", "debug", env_id=self.env_id, kind=kind, **kw)["result"]

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


# ---- Display -------------------------------------------------------------------------------------
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
    p = argparse.ArgumentParser(description="Teleop client, step 1: viewer + input test + latency.")
    p.add_argument("--host", default="localhost", help="Local end of the ssh tunnel.")
    p.add_argument("--port", type=int, default=8102, help="Local port of the tunnel (server_port of the YAML).")
    p.add_argument("--seed", type=int, default=100, help="Reset seed of the first episode; R uses seed + 1, ...")
    p.add_argument("--hz", type=float, default=15.0,
                   help="Loop rate. 15 = the control rate of the policy. 0 = as fast as the server allows "
                        "(to measure the raw speed).")
    p.add_argument("--scale", type=int, default=3, help="Camera enlargement factor (224 px x scale).")
    p.add_argument("--max-steps", type=int, default=150, help="Episode length shown in the HUD (YAML max_steps_per_episode).")
    p.add_argument("--max-seconds", type=float, default=0.0, help="Quit by itself after this time and print the "
                                                                  "latency summary (0 = no limit).")
    args = p.parse_args()

    import pygame
    pygame.init()
    size = 224 * args.scale
    hud_height = 190
    screen = pygame.display.set_mode((max(2 * size, 1000), size + hud_height))
    pygame.display.set_caption("FORGE PegInsert -- teleop client (step 1)")
    font = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 18)
    big_font = pygame.font.SysFont("consolas,dejavusansmono,couriernew", 28, bold=True)
    clock = pygame.time.Clock()

    env = RemoteEnv(args.host, args.port)
    prompt = env.create_env()
    geom = env.debug("task_geometry")
    print(f"connected to {env.uri}   task: {prompt!r}")
    print(f"hole depth {geom['hole_depth'] * 1000:.0f} mm   success: xy < {geom['success_xy'] * 1000:.1f} mm, "
          f"z < {geom['success_z'] * 1000:.1f} mm")

    seed = args.seed
    obs = env.reset(seed)
    quat0 = env.debug("fingertip_pose")["quat"]           # orientation to hold in step 2
    print(f"observation keys: {sorted(obs.keys())}")
    print("episode", seed, "started.  R = new episode, Tab = capture the mouse, Esc = quit.")

    step_count, over, success, grabbed = 0, False, False, False
    mouse_total = np.zeros(2)
    loop_ms = collections.deque(maxlen=60)
    t_start = time.perf_counter()
    last_print = t_start
    running = True
    while running:
        t_loop = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    seed += 1
                    obs = env.reset(seed)
                    step_count, over, success = 0, False, False
                    print("episode", seed, "started.")
                elif event.key == pygame.K_TAB:
                    grabbed = not grabbed
                    pygame.event.set_grab(grabbed)
                    pygame.mouse.set_visible(not grabbed)
                    pygame.mouse.get_rel()                   # forget the jump caused by the capture
        keys = pygame.key.get_pressed()
        dx, dy = pygame.mouse.get_rel()
        mouse_total += (dx, dy)

        if not over:
            # Exactly the calls step 2 will make at each control step.
            q_meas = np.asarray(obs["observation/joint_position"], dtype=np.float64)
            env.debug("ik_joint_target", timer="ik", delta_pos=[0.0, 0.0, 0.0], target_quat=quat0)   # result unused here
            env.step(np.concatenate([q_meas, [GRIPPER_CLOSED]]))
            obs = env.get_observation()
            over, success = env.get_info()
            step_count += 1
            if step_count >= args.max_steps:
                over = True

        # -- draw --
        screen.fill(BACKGROUND)
        screen.blit(to_surface(pygame, obs["observation/exterior_image_1_left"], args.scale), (0, 0))
        screen.blit(to_surface(pygame, obs["observation/wrist_image_left"], args.scale), (size, 0))
        offset_mm = np.asarray(obs["extra/peg_socket_offset"], dtype=np.float64) * 1000.0
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
        status = "SUCCESS" if success else ("episode over -- press R" if over else "running")
        loop_hz = 1000.0 / statistics.fmean(loop_ms) if loop_ms else float("nan")
        y = draw_text(screen, font, 10, y,
                      f"seed {seed}   step {step_count}/{args.max_steps}   {status}   loop {loop_hz:5.1f} Hz "
                      f"(target {args.hz:g})   mouse {'CAPTURED' if grabbed else 'free'}",
                      GOOD if success else (WARN if over else TEXT))
        # Round trip times (ms): mean / 95th percentile. Their sum is the minimum time of one control step.
        parts = [f"{name} {env.mean_ms(name):.1f}/{env.p95_ms(name):.1f}" for name in ("ik", "step", "obs", "info")]
        total = sum(env.mean_ms(n) for n in ("ik", "step", "obs", "info"))
        y = draw_text(screen, font, 10, y, "ms mean/p95:  " + "  ".join(parts) + f"  | sum {total:.1f}",
                      WARN if total > 1000.0 / max(args.hz, 1e-6) else TEXT)
        pressed = [name for name, key in (("W", pygame.K_w), ("S", pygame.K_s), ("A", pygame.K_a), ("D", pygame.K_d),
                                          ("Shift", pygame.K_LSHIFT), ("Ctrl", pygame.K_LCTRL),
                                          ("Space", pygame.K_SPACE)) if keys[key]]
        y = draw_text(screen, font, 10, y, f"input test:  mouse d=({dx:+4d},{dy:+4d})  sum=({mouse_total[0]:+6.0f},"
                                           f"{mouse_total[1]:+6.0f})   keys: {' '.join(pressed) or '-'}", DIM)
        draw_text(screen, font, 10, y, "R new episode   Tab capture mouse   Esc quit   (step 1: the arm holds its pose)", DIM)
        pygame.display.flip()

        # -- pace the loop --
        if args.hz > 0:
            clock.tick(args.hz)
        loop_ms.append((time.perf_counter() - t_loop) * 1000.0)
        now = time.perf_counter()
        if now - last_print > 5.0:
            last_print = now
            print(f"  loop {loop_hz:5.1f} Hz   " + "   ".join(f"{n} {env.mean_ms(n):.1f} ms" for n in ("ik", "step", "obs", "info")))
        if args.max_seconds and now - t_start > args.max_seconds:
            running = False

    print("\nround trips (ms)        mean     p95")
    for name in ("ik", "step", "obs", "info"):
        print(f"  {name:<18s} {env.mean_ms(name):7.1f} {env.p95_ms(name):7.1f}")
    total = sum(env.mean_ms(n) for n in ("ik", "step", "obs", "info"))
    print(f"  one control step   {total:7.1f}   -> at most {1000.0 / total:.1f} Hz")
    env.close()
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
