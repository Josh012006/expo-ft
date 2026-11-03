"""Tactile / force input of the residual actor and the critic (EXPOLearner only).

What this is
    An optional, fixed-size vector built from signals of the FORGE server observation, selected by THREE task-YAML
    fields (the single place where they are read is `AuxSpec.from_config`):

        use_tactile   bool                    the taxels
        use_force     bool                    the force signals
        force_mode    "measured" | "proxy"    where the force comes from (only meaningful with use_force)

    Both bools off (the default) = the task alone: nothing below is built, stored or fed to any network, and every
    shape is the one of the code before this module existed.

    Signals (keys of the server observation):

        use_force, force_mode "measured":
            extra/ft_force         (3,)       wrist force sensor, N
            extra/gripper_effort   scalar     PD torque demanded of the Robotiq finger joint, N.m (Robotiq robots only)
        use_tactile:
            extra/tactile          (2, 7, 4)  net contact force of each taxel zone, N (robot "robotiq_2f85_tsf" only)
        use_force, force_mode "proxy":
            routed but NOT implemented yet (the force is then estimated instead of read, arXiv 2610.23849). It raises
            NotImplementedError when the AuxSpec is created, i.e. before anything is built or any GPU time is spent.

    Layout of the vector, only the enabled parts present, in this order:
        [ft_force (3) | gripper_effort (1) | tactile flattened in C order (56)]      dim = 4*use_force + 56*use_tactile

Where it goes
    ONLY into the residual actor and the critic, as extra columns of `critic_states` (state of the pi0.5 base policy
    followed by this vector). The `state` that the frozen base policy receives is never touched: the base was
    fine-tuned without touch.

Where it is built
    From the raw observation dict, by `AuxSpec.build`, in exactly two places: PiReplayBuffer.insert (training data,
    demonstrations included) and EXPOLearner.sample_actions (acting). Nothing else computes it, so the two cannot drift
    apart. A missing key raises (a zero vector would be a silent, wrong "no contact" signal).

Scaling
    A single parameter-free map, applied to every entry: sign(x) * log1p(|x|). It compresses the large forces of a
    hard contact without a clip or a hand-tuned scale, keeps the sign, and maps 0 to 0. The network then applies its
    own Dense -> LayerNorm -> tanh projection (same as for the joint state).
"""

from dataclasses import dataclass

import numpy as np

FT_FORCE_KEY = "extra/ft_force"
TACTILE_KEY = "extra/tactile"
GRIPPER_EFFORT_KEY = "extra/gripper_effort"

FT_FORCE_SHAPE = (3,)
GRIPPER_EFFORT_SHAPE = (1,)
TACTILE_SHAPE = (2, 7, 4)
FORCE_DIM = int(np.prod(FT_FORCE_SHAPE)) + int(np.prod(GRIPPER_EFFORT_SHAPE))   # 3 + 1 = 4
TACTILE_DIM = int(np.prod(TACTILE_SHAPE))                                        # 2 * 7 * 4 = 56

FORCE_MODES = ("measured", "proxy")
IMPLEMENTED_FORCE_MODES = ("measured",)

# Task-YAML / model-config field names, shared by every reader (single source, no string typed twice).
CONFIG_FIELDS = ("use_tactile", "use_force", "force_mode")


def _check_shape(name, value, shape):
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")


@dataclass(frozen=True)
class AuxSpec:
    """Which extra signals the residual actor and the critic see. Immutable and hashable (it is a static field of the
    jitted agent)."""

    use_tactile: bool = False
    use_force: bool = False
    force_mode: str = "measured"

    def __post_init__(self):
        for name in ("use_tactile", "use_force"):
            if not isinstance(getattr(self, name), bool):
                # A YAML string such as "false" would be truthy: refuse instead of silently turning a signal on.
                raise TypeError(f"{name} must be a bool (true / false), got {getattr(self, name)!r}")
        if self.force_mode not in FORCE_MODES:
            raise ValueError(f"force_mode must be one of {FORCE_MODES}, got {self.force_mode!r}")
        if not self.use_force and self.force_mode != "measured":
            raise ValueError(
                f"force_mode: {self.force_mode!r} has no effect while use_force is false: set use_force: true, "
                f"or remove force_mode."
            )
        if self.use_force and self.force_mode not in IMPLEMENTED_FORCE_MODES:
            raise NotImplementedError(
                f"force_mode: {self.force_mode!r} is routed but not implemented yet (the force proxy, "
                f"arXiv 2610.23849). Implemented: {IMPLEMENTED_FORCE_MODES}."
            )

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config) -> "AuxSpec":
        """From the task YAML (SimpleNamespace) or the model config (ConfigDict): the same three fields in both. A field
        that is absent means off / "measured" (configs of the other learners, old YAMLs)."""
        return cls(
            use_tactile=getattr(config, "use_tactile", False),
            use_force=getattr(config, "use_force", False),
            force_mode=getattr(config, "force_mode", "measured"),
        )

    def to_config_kwargs(self) -> dict:
        return {"use_tactile": self.use_tactile, "use_force": self.use_force, "force_mode": self.force_mode}

    # ------------------------------------------------------------------ description
    @property
    def enabled(self) -> bool:
        return self.use_tactile or self.use_force

    @property
    def dim(self) -> int:
        return (FORCE_DIM if self.use_force else 0) + (TACTILE_DIM if self.use_tactile else 0)

    @property
    def keys(self) -> tuple:
        """Observation keys that `build` reads, in layout order."""
        keys = ()
        if self.use_force:                                       # force_mode "measured" (the only implemented one)
            keys += (FT_FORCE_KEY, GRIPPER_EFFORT_KEY)
        if self.use_tactile:
            keys += (TACTILE_KEY,)
        return keys

    def describe(self) -> str:
        if not self.enabled:
            return "none (task only)"
        parts = []
        if self.use_force:
            parts.append(f"force[{self.force_mode}] (ft_force 3 + gripper_effort 1)")
        if self.use_tactile:
            parts.append("tactile (2x7x4)")
        return " + ".join(parts) + f" = {self.dim} values"

    # ------------------------------------------------------------------ data
    def build(self, observation) -> np.ndarray:
        """Raw environment observation (dict) -> (dim,) float32 vector, see the module docstring.

        Raises instead of filling in zeros: the ways to get here without the signals are a robot without taxels /
        gripper effort (the server only sends `extra/tactile` for `robotiq_2f85_tsf` and `extra/gripper_effort` for
        Robotiq robots) and demonstrations that were recorded without them.
        """
        if not self.enabled:
            raise RuntimeError("AuxSpec.build called while use_tactile and use_force are both false")
        missing = [k for k in self.keys if k not in observation]
        if missing:
            raise KeyError(
                f"{self.describe()} is enabled but the observation has no {missing}. These keys come from the "
                f"environment server: the Robotiq robot is required for extra/gripper_effort and robot "
                f"'robotiq_2f85_tsf' for extra/tactile. Demonstrations inserted into a replay buffer must carry "
                f"them as well (the .npz files record them when the server sends them)."
            )

        parts = []
        if self.use_force:
            force = np.asarray(observation[FT_FORCE_KEY], dtype=np.float32)
            effort = np.asarray(observation[GRIPPER_EFFORT_KEY], dtype=np.float32).reshape(-1)
            _check_shape(FT_FORCE_KEY, force, FT_FORCE_SHAPE)
            _check_shape(GRIPPER_EFFORT_KEY, effort, GRIPPER_EFFORT_SHAPE)
            parts += [force, effort]
        if self.use_tactile:
            tactile = np.asarray(observation[TACTILE_KEY], dtype=np.float32)
            _check_shape(TACTILE_KEY, tactile, TACTILE_SHAPE)
            parts.append(tactile.reshape(-1))

        raw = np.concatenate(parts)
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"Non-finite value in the tactile / force observation: {raw}")
        return (np.sign(raw) * np.log1p(np.abs(raw))).astype(np.float32)

    def dummy_observations(self) -> dict:
        """Zero-valued `extra/*` entries of the right shapes, for the synthetic transition that only initialises
        shapes (expo_ft.env.droid_utils.make_dummy_transition). Never read as data."""
        zeros = {
            FT_FORCE_KEY: np.zeros(FT_FORCE_SHAPE, dtype=np.float32),
            GRIPPER_EFFORT_KEY: np.float32(0.0),
            TACTILE_KEY: np.zeros(TACTILE_SHAPE, dtype=np.float32),
        }
        return {k: zeros[k] for k in self.keys}
