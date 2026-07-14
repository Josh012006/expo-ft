"""TEMPORARY passthrough.

expo_ft.py has been renamed to expo_ft_old.py so the current (MSE-scalar
critic, argmax-over-candidates) implementation stays available for
comparison/rollback while a new categorical (XQC/XQCfD-style, bounded
support) critic architecture is built in its place.

This file exists purely so that `expo_ft/agents/alg/__init__.py` — and
therefore the whole `expo_ft.agents.alg` package (PPO, GRPO, SAC, BC,
eval_policy.py, eval_droid_policy.py all import from this package) — keeps
working unchanged in the meantime. It re-exports everything the old module
provided, unmodified.

Once the categorical-critic rewrite is ready, its code replaces the content
of THIS file (expo_ft.py) directly — nothing importing "expo_ft.agents.alg.
expo_ft" needs to change. To roll back to the old architecture at any point,
either:
  - swap the two files' contents back, or
  - point train_pi_robo.py's EXPOLearner import at expo_ft_old instead.
"""

from expo_ft.agents.alg.expo_ft_old import *  # noqa: F401,F403
from expo_ft.agents.alg.expo_ft_old import (  # noqa: F401
    EXPOLearner,
    load_agent,
    restore_checkpoint,
    save_checkpoint,
)
