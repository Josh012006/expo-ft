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
"""
import logging
import traceback
from typing import Any, Protocol

import websockets.sync.server
from openpi_client import msgpack_numpy

logging.basicConfig(level=logging.INFO)


class Backend(Protocol):
    """Contract a simulation (or real robot) backend must implement."""

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

    One connection is handled fully before the next is accepted, which is
    fine here: Isaac Sim's stepping is itself single-threaded and blocking,
    so nothing is gained from concurrent connections, and this keeps the
    simulation main-thread requirement trivially satisfied.
    """

    def __init__(self, backend: Backend, host: str = "0.0.0.0", port: int = 8102):
        self.backend = backend
        self.host = host
        self.port = port
        self._packer = msgpack_numpy.Packer()

    def _handle_request(self, request: dict) -> dict:
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

            return {"status": "error", "message": f"Unknown operation: {op}"}
        except Exception as e:  # noqa: BLE001 — must always return a response, never crash the loop
            logging.error("Operation %s failed:\n%s", op, traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def _connection_handler(self, ws: "websockets.sync.server.ServerConnection"):
        logging.info("Client connected.")
        try:
            for raw in ws:
                request = msgpack_numpy.unpackb(raw)
                response = self._handle_request(request)
                ws.send(self._packer.pack(response))
        except websockets.exceptions.ConnectionClosed:
            logging.info("Client disconnected.")

    def serve_forever(self):
        logging.info("Env server listening on ws://%s:%s", self.host, self.port)
        with websockets.sync.server.serve(
            self._connection_handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            server.serve_forever()
