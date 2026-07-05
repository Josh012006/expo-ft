"""
Thin wrapper around mani_skill.trajectory.replay_trajectory that applies the
PickCube visible-goal patch (expo_ft/env/patches.py) BEFORE running the
normal replay_trajectory CLI logic, unchanged otherwise.

Why this exists: replay_trajectory runs in its own subprocess (invoked via
`python -m mani_skill.trajectory.replay_trajectory ...`), so a monkeypatch
applied only inside expo_ft's own process (e.g. in maniskill_env.py) never
reaches it. This script re-applies the same patch inside THAT process instead.

Harmless no-op for tasks without a goal_site (StackCube, PushCube) — safe to
use as a drop-in replacement everywhere, not just for PickCube.

Usage: identical CLI args as the original tool, e.g.
    python scripts/replay_trajectory_patched.py \\
        --traj-path demos/PickCube-v1/motionplanning/trajectory.h5 \\
        --save-traj -o rgb -c pd_joint_delta_pos -b physx_cpu
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from expo_ft.env.patches import patch_pickcube_visible_goal
patch_pickcube_visible_goal()

from mani_skill.trajectory.replay_trajectory import main, parse_args

if __name__ == "__main__":
    main(parse_args())
