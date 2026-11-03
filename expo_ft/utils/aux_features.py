"""Tactile / force / gripper-effort input of the residual actor and the critic (EXPOLearner only).

What this is
    A fixed-size vector built from three signals of the FORGE server observation:

        extra/ft_force         (3,)       wrist force sensor, N
        extra/tactile          (2, 7, 4)  net contact force of each taxel zone, N (needs robot "robotiq_2f85_tsf")
        extra/gripper_effort   scalar     PD torque demanded of the Robotiq finger joint, N.m (Robotiq robots only)

    Layout of the vector (AUX_DIM = 60): [ft_force (3) | tactile flattened in C order (56) | gripper_effort (1)].

Where it goes
    ONLY into the residual actor and the critic, as extra columns of `critic_states` (state of the pi0.5 base policy
    followed by this vector). The `state` that the frozen base policy receives is never touched: the base was
    fine-tuned without touch.

Where it is built
    From the raw observation dict, by `build_aux_features`, in exactly two places: PiReplayBuffer.insert (training data)
    and EXPOLearner.sample_actions (acting). Nothing else computes it, so the two cannot drift apart.

Scaling
    A single parameter-free map, applied to every entry: sign(x) * log1p(|x|). It compresses the large forces of a
    hard contact without a clip or a hand-tuned scale, keeps the sign, and maps 0 to 0. The network then applies its
    own Dense -> LayerNorm -> tanh projection (same as for the joint state).
"""

import numpy as np

FT_FORCE_KEY = "extra/ft_force"
TACTILE_KEY = "extra/tactile"
GRIPPER_EFFORT_KEY = "extra/gripper_effort"
AUX_KEYS = (FT_FORCE_KEY, TACTILE_KEY, GRIPPER_EFFORT_KEY)

FT_FORCE_SHAPE = (3,)
TACTILE_SHAPE = (2, 7, 4)
GRIPPER_EFFORT_SHAPE = (1,)
AUX_DIM = (
    int(np.prod(FT_FORCE_SHAPE)) + int(np.prod(TACTILE_SHAPE)) + int(np.prod(GRIPPER_EFFORT_SHAPE))
)  # 3 + 56 + 1 = 60


def build_aux_features(observation) -> np.ndarray:
    """Raw environment observation (dict) -> (AUX_DIM,) float32 vector, see the module docstring.

    Raises instead of filling in zeros: a zero vector would be a silent, wrong "no contact" signal. The two ways to get
    here without the signals are a robot without taxels / gripper effort (the server only sends `extra/tactile` for
    `robotiq_2f85_tsf` and `extra/gripper_effort` for Robotiq robots) and demonstrations that were recorded without them.
    """
    missing = [k for k in AUX_KEYS if k not in observation]
    if missing:
        raise KeyError(
            f"use_tactile_aux is enabled but the observation has no {missing}. These keys come from the environment "
            f"server: robot 'robotiq_2f85_tsf' is required (taxels and gripper effort). Demonstrations inserted into a "
            f"replay buffer must carry them as well."
        )

    force = np.asarray(observation[FT_FORCE_KEY], dtype=np.float32)
    tactile = np.asarray(observation[TACTILE_KEY], dtype=np.float32)
    effort = np.asarray(observation[GRIPPER_EFFORT_KEY], dtype=np.float32).reshape(-1)
    for name, value, shape in (
        (FT_FORCE_KEY, force, FT_FORCE_SHAPE),
        (TACTILE_KEY, tactile, TACTILE_SHAPE),
        (GRIPPER_EFFORT_KEY, effort, GRIPPER_EFFORT_SHAPE),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {shape}")

    raw = np.concatenate([force, tactile.reshape(-1), effort])
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"Non-finite value in the tactile / force / gripper-effort observation: {raw}")
    return (np.sign(raw) * np.log1p(np.abs(raw))).astype(np.float32)


def dummy_aux_observations() -> dict:
    """Zero-valued `extra/*` entries of the right shapes, for the synthetic transition that only initialises shapes
    (expo_ft.env.droid_utils.make_dummy_transition). Never read as data."""
    return {
        FT_FORCE_KEY: np.zeros(FT_FORCE_SHAPE, dtype=np.float32),
        TACTILE_KEY: np.zeros(TACTILE_SHAPE, dtype=np.float32),
        GRIPPER_EFFORT_KEY: np.float32(0.0),
    }
