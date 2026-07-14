"""Generic checkpoint save/restore logic, shared across every learner
(EXPOLearner, PPOLearner, GRPOLearner, SACLearner, BCLearner).

Each learner's actual parameter pytree structure is different (different
field names: batch_encoder/residual_actor/temp/critic/actor_train_state for
EXPOLearner, batch_encoder/actor/value for PPOLearner, etc.), so the
per-learner `_split_params`/`_merge_params` functions can't be shared — but
the *mechanics* around them (call split, hand {"agent": ..., "params": ...}
to orbax, call merge on the way back) are identical every time. That
mechanics-only part lives here, once, instead of being copy-pasted five
times.

This file is deliberately kept stable and free of any single learner's
internals — nothing here should ever need to change when a specific
architecture (e.g. expo_ft.py) is being rewritten or swapped out, which is
exactly what makes it safe for expo_ft/agents/alg/__init__.py to depend on.
"""

from typing import Any, Callable, Tuple

import orbax.checkpoint as ocp


def make_checkpoint_fns(
    split_fn: Callable[[Any], Tuple[Any, dict]],
    merge_fn: Callable[[Any, dict], Any],
):
    """Build a (restore_checkpoint, save_checkpoint) pair for one learner.

    Args:
        split_fn: agent -> (agent_with_params_zeroed_out, params_dict).
            Each learner defines its own — pulls the trainable params out of
            whichever TrainStates it has so orbax can checkpoint them
            separately from the rest of the (non-pytree) agent structure.
        merge_fn: (agent_with_params_zeroed_out, params_dict) -> agent.
            The inverse of split_fn, used after restoring from disk.

    Returns:
        (restore_checkpoint, save_checkpoint) functions with the exact same
        call signature every learner file already used individually.
    """

    def restore_checkpoint(checkpoint_manager: ocp.CheckpointManager, agent: Any, step: int | None = None):
        agent, params = split_fn(agent)
        restored = checkpoint_manager.restore(
            step,
            items={"agent": agent, "params": params},
        )
        return merge_fn(restored["agent"], restored["params"])

    def save_checkpoint(checkpoint_manager: ocp.CheckpointManager, agent: Any, step: int):
        agent, params = split_fn(agent)
        checkpoint_manager.save(step, {"agent": agent, "params": params})

    return restore_checkpoint, save_checkpoint
