"""Generic synchronous WebSocket environment server.

Speaks the exact wire protocol expected by expo_ft.env.env_client.EnvClient
(the client already used for the real DROID robot): one persistent
connection, msgpack_numpy-packed {"operation": ..., **kwargs} requests,
{"status": "ok"|"error", ...} responses.

This module knows NOTHING about Isaac Lab or FORGE — it only dispatches to
a `Backend` object that implements the five operations. That separation is
deliberate: swapping this file for an asyncio-based server later (if the
sync design turns out to be a bottleneck) should never require touching
forge_env_patched.py or forge_server.py.

Threading note (the actual reason for the queue below): websockets.sync
handles each connection on its own thread — that's the whole point of the
"sync" variant, it lets connection handlers be written as plain blocking
functions. Isaac Sim/Kit, however, must be driven from the thread that
created AppLauncher; calling env.reset()/env.step() from a connection
thread doesn't raise, it silently hangs waiting on something only the
owning thread can service. So network I/O stays on its connection threads,
but every Backend call is handed off to, and executed exclusively on,
whichever thread calls serve_forever() (the main thread in forge_server.py).
For num_envs=1 / one connection at a time, this costs nothing.
"""
import logging
import queue
import threading
import traceback
from typing import Any, Protocol

import websockets.sync.server
from openpi_client import msgpack_numpy

logging.basicConfig(level=logging.INFO)


class Backend(Protocol):
    """Contract a simulation (or real robot) backend must implement.
    Every method here is guaranteed to run on the thread that called
    WsEnvServer.serve_forever() — see the threading note above."""

    def create_env(self, request: dict) -> tuple[str, str]:
        """Return (env_id, task_description)."""
        ...

    def reset(self, env_id: str, seed: int | None) -> tuple[dict, bool]:
        """Return (observation, done)."""
        ...

    def step(self, env_id: str, action) -> tuple[Any, str]:
        """Return (real_executed_action, action_type)."""
        ...

    def get_observation(self, env_id: str) -> dict:
        ...

    def get_info_for_step(self, env_id: str) -> tuple[bool, bool, float, float]:
        """Return (done, success, reward, mask)."""
        ...


class WsEnvServer:
    """Synchronous WebSocket server dispatching to a Backend.

    Network I/O runs on connection threads spawned by websockets.sync;
    Backend calls are marshalled onto serve_forever()'s calling thread via
    a work queue (see module docstring).
    """

    def __init__(self, backend: Backend, host: str = "0.0.0.0", port: int = 8102):
        self.backend = backend
        self.host = host
        self.port = port
        self._packer = msgpack_numpy.Packer()
        self._work_queue: "queue.Queue[tuple[dict, queue.Queue]]" = queue.Queue()

    def _dispatch(self, request: dict) -> dict:
        """Runs on the main thread only — the only place Backend is touched."""
        op = request.get("operation")
        try:
            if op == "create_env":
                env_id, task_description = self.backend.create_env(request)
                return {"status": "ok", "env_id": env_id, "task_description": task_description}

            if op == "reset":
                obs, done = self.backend.reset(request["env_id"], request.get("seed"))
                return {"status": "ok", "observation": obs, "done": done}

            if op == "step":
                real_action, action_type = self.backend.step(request["env_id"], request["action"])
                return {"status": "ok", "action": real_action, "action_type": action_type}

            if op == "get_observation":
                obs = self.backend.get_observation(request["env_id"])
                return {"status": "ok", "observation": obs}

            if op == "get_info_for_step":
                done, success, reward, mask = self.backend.get_info_for_step(request["env_id"])
                return {"status": "ok", "done": done, "success": success, "reward": reward, "mask": mask}

            if op == "debug":
                # Optional, diagnostic-only escape hatch — NOT part of the
                # stable protocol env_client.py exposes publicly (that one
                # must stay real-robot-compatible). Backends that don't
                # implement debug() simply don't support it.
                debug_fn = getattr(self.backend, "debug", None)
                if debug_fn is None:
                    return {"status": "error", "message": "backend has no debug() method"}
                return {"status": "ok", "result": debug_fn(request)}

            return {"status": "error", "message": f"Unknown operation: {op}"}
        except Exception as e:  # noqa: BLE001 — must always return a response, never crash the loop
            logging.error("Operation %s failed:\n%s", op, traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def _handle_request(self, request: dict) -> dict:
        """Runs on a connection thread: hand the request to the main thread
        and block until it's actually been executed there."""
        result_queue: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        self._work_queue.put((request, result_queue))
        return result_queue.get()

    def _connection_handler(self, ws: "websockets.sync.server.ServerConnection"):
        logging.info("Client connected.")
        try:
            for raw in ws:
                request = msgpack_numpy.unpackb(raw)
                response = self._handle_request(request)
                ws.send(self._packer.pack(response))
        except websockets.exceptions.ConnectionClosed:
            logging.info("Client disconnected.")

    def _run_ws_server(self):
        logging.info("Env server listening on ws://%s:%s", self.host, self.port)
        with websockets.sync.server.serve(
            self._connection_handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            server.serve_forever()

    def serve_forever(self):
        """Starts the WebSocket accept loop on a background thread, then
        pumps the work queue on THIS thread forever. Call this from the
        same thread that created AppLauncher / the env."""
        threading.Thread(target=self._run_ws_server, daemon=True).start()
        while True:
            request, result_queue = self._work_queue.get()
            result_queue.put(self._dispatch(request))
