"""Runtime monkeypatches for ManiSkill task environments.

These edit environment CLASSES (not just instances) at import time, so they
apply uniformly regardless of which code path creates the env — our own
ManiSkillEnvWrapper, or a separate ManiSkill CLI tool like replay_trajectory
run in its own subprocess (see scripts/replay_trajectory_patched.py, which
needs the patch applied there too since it's a different process).
"""


def patch_pickcube_visible_goal():
    """Make PickCube-v1's goal marker sphere visible to the agent's sensor
    cameras.

    By default ManiSkill hides it (`self._hidden_objects.append(self.goal_site)`
    in PickCubeEnv._load_scene) so it never appears in sensor observations —
    fine for state-based/privileged RL (goal_pos is still exposed numerically
    via obs['extra']), but our pipeline is image+proprioception only (matching
    real DROID robots, no privileged state channel), so the goal is otherwise
    completely unobservable to the policy. Removing it from _hidden_objects
    lets the agent's camera actually see it, matching how PushCube's
    equivalent goal_region is already left visible upstream.

    Safe no-op if PickCubeEnv isn't installed/importable. Idempotent — safe to
    call multiple times (e.g. once per process that imports this module).
    """
    try:
        from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
    except ImportError:
        return

    if getattr(PickCubeEnv, "_expo_ft_visible_goal_patched", False):
        return

    _original_load_scene = PickCubeEnv._load_scene

    def _load_scene_visible_goal(self, options):
        _original_load_scene(self, options)
        goal_site = getattr(self, "goal_site", None)
        if goal_site is not None and goal_site in self._hidden_objects:
            self._hidden_objects.remove(goal_site)

    PickCubeEnv._load_scene = _load_scene_visible_goal
    PickCubeEnv._expo_ft_visible_goal_patched = True
