"""Action frame of the model versus the one of the environment (task-YAML key `chunk_action_reference`).

Two frames exist:

    environment   what env.step() receives and what a demonstration records: for FORGE, 7 ABSOLUTE joint targets (rad)
                  + the gripper command (0 open, 1 closed).
    model         what pi05_droid_jointpos outputs and what the residual actor and the critic work with: a chunk of
                  OFFSETS from the joint state at the START OF THE CHUNK (openpi's DeltaActions encoding) for the 7
                  joints, the gripper stays absolute. The weights were trained that way and the openpi config used for
                  RL (..._joint_state) has no AbsoluteActions in its outputs, so the policy returns offsets.

`chunk_action_reference` says how to go from one to the other:

    null (default)         the two frames are the same (ManiSkill: pd_joint_delta_pos, per-step deltas). Nothing here runs.
    "chunk_start_state"    model = environment - joint_state(at the step the chunk is anchored to), on the 7 joints.

Where the conversion happens, and nowhere else:

    environment <- model   `chunk_to_env_actions`, once per chunk right after agent.sample_actions(...), with the
                           observation the chunk was planned from (scripts/eval_policy.py, train_pi_robo.py).
    model <- environment   PiReplayBuffer.insert, for every stored step, demonstrations and online transitions alike:
                           the chunk of row r holds (action[r+k] - joint_state[r]) for the 7 joints, normalized, which is
                           exactly what DeltaActions + Normalize give the SFT demos and the candidates of the base policy.

Mid-chunk rows: while a chunk is being executed the next rows are anchored to THEIR OWN joint state, like every demo row
(the chunk "from this state onwards"), not to the plan-time state of the chunk they happen to belong to.
"""

import numpy as np

CHUNK_START_STATE = "chunk_start_state"
SUPPORTED_REFERENCES = (None, CHUNK_START_STATE)
N_ANCHORED_DIMS = 7        # the 7 arm joints; dimension 8 (gripper) is absolute in both frames (openpi make_bool_mask(7, -1))


def validate_reference(reference):
    """Return `reference` if it is a supported value, raise otherwise (a typo must not silently mean "no conversion")."""
    if reference not in SUPPORTED_REFERENCES:
        raise ValueError(f"chunk_action_reference must be one of {SUPPORTED_REFERENCES}, got {reference!r}")
    return reference


def validate_task(cfg):
    """Check the task YAML fields that the conversion relies on. Returns the reference (None = no conversion)."""
    reference = validate_reference(getattr(cfg, "chunk_action_reference", None))
    if reference is None:
        return None
    if getattr(cfg, "use_cartesian_state", False):
        raise ValueError("chunk_action_reference: chunk_start_state needs a joint state (use_cartesian_state: false)")
    if cfg.state_obs_dim < N_ANCHORED_DIMS or cfg.output_action_dim < N_ANCHORED_DIMS:
        raise ValueError(
            f"chunk_action_reference: chunk_start_state anchors {N_ANCHORED_DIMS} joints, but state_obs_dim="
            f"{cfg.state_obs_dim} and output_action_dim={cfg.output_action_dim}"
        )
    return reference


def anchor_from_observation(observation, state_obs_key) -> np.ndarray:
    """(7,) float32: the joint positions of an observation, i.e. what openpi's DeltaActions subtracts (state[:7])."""
    if state_obs_key is None or state_obs_key not in observation:
        raise KeyError(f"chunk_action_reference needs the joint state under {state_obs_key!r}; the observation has "
                       f"{sorted(observation)}")
    anchor = np.asarray(observation[state_obs_key], dtype=np.float32).reshape(-1)
    if anchor.shape[0] < N_ANCHORED_DIMS:
        raise ValueError(f"{state_obs_key} has {anchor.shape[0]} values, expected at least {N_ANCHORED_DIMS}")
    return anchor[:N_ANCHORED_DIMS]


def chunk_to_env_actions(action_chunk, observation, reference, state_obs_key) -> np.ndarray:
    """Model chunk (H, action_dim) -> chunk the environment can execute. `observation` is the one the chunk was
    planned from. A copy: the input is not modified. reference None returns the chunk unchanged (as an array)."""
    chunk = np.array(action_chunk, dtype=np.float32, copy=True)
    if reference is None:
        return chunk
    validate_reference(reference)
    chunk[:, :N_ANCHORED_DIMS] += anchor_from_observation(observation, state_obs_key)
    return chunk


def env_to_model_actions(env_actions, anchor) -> np.ndarray:
    """Environment actions (..., action_dim) -> model frame, for the anchor (7,) of the state they are expressed
    relative to. The inverse of chunk_to_env_actions for one anchor. A copy."""
    actions = np.array(env_actions, dtype=np.float32, copy=True)
    actions[..., :N_ANCHORED_DIMS] -= np.asarray(anchor, dtype=np.float32)
    return actions
