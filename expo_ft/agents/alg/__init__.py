from expo_ft.agents.alg.agent import AgentLearner, initialize_checkpoint_dir
from expo_ft.agents.alg.bc import BCLearner
from expo_ft.data.replay_buffer import (
    save_replay_buffer_transition,
    restore_replay_buffer,
)

# NOTE: EXPOLearner, restore_checkpoint, and save_checkpoint used to be
# re-exported here from expo_ft.agents.alg.expo_ft. Nothing in the codebase
# actually imported them this way (every real consumer — train_pi_robo.py,
# eval_policy.py, eval_droid_policy.py — imports directly from the specific
# algo submodule instead, e.g. `from expo_ft.agents.alg.expo_ft import
# load_agent`), and with five learners now (EXPOLearner, PPOLearner,
# GRPOLearner, SACLearner, BCLearner) "the" restore_checkpoint/save_checkpoint
# at package level is ambiguous anyway. Removed so this package's __init__
# doesn't hard-depend on expo_ft.py specifically — that file is the one most
# likely to be under active rewrite (e.g. the XQC categorical-critic work),
# and a broken/incomplete expo_ft.py should not take down PPO/GRPO/SAC/BC
# (or this package's own import) along with it.
# If you need EXPOLearner/restore_checkpoint/save_checkpoint, import them
# directly: `from expo_ft.agents.alg.expo_ft import EXPOLearner, ...`.

__all__ = [
    "AgentLearner",
    "BCLearner",
    "initialize_checkpoint_dir",
    "save_replay_buffer_transition",
    "restore_replay_buffer",
]
