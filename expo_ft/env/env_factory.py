"""
Environment wrapper factory for EXPO-FT.
Selects the correct wrapper based on cfg.env_wrapper.

Supported values:
    "maniskill"  → ManiSkillEnvWrapper  (default)
    "libero"     → LiberoEnvWrapper
    "robocasa"   → RoboCasaEnvWrapper
"""


def make_env_wrapper(env_creation_request: dict, cfg=None):
    """
    Factory function — returns the correct env wrapper based on cfg.env_wrapper.

    Args:
        env_creation_request: dict with keys: example_action, env_usage, video_dir
        cfg: task config loaded from YAML

    Returns:
        An env wrapper instance with the ManiSkillEnvWrapper interface.
    """
    env_wrapper = getattr(cfg, 'env_wrapper', 'maniskill')

    if env_wrapper == 'maniskill':
        from expo_ft.env.maniskill_env import ManiSkillEnvWrapper
        return ManiSkillEnvWrapper(env_creation_request, cfg)

    elif env_wrapper == 'libero':
        from expo_ft.env.libero_env import LiberoEnvWrapper
        return LiberoEnvWrapper(env_creation_request, cfg)

    elif env_wrapper == 'robocasa':
        from expo_ft.env.robocasa_env import RoboCasaEnvWrapper
        return RoboCasaEnvWrapper(env_creation_request, cfg)

    else:
        raise ValueError(
            f"Unknown env_wrapper: '{env_wrapper}'. "
            f"Supported: 'maniskill', 'libero', 'robocasa'"
        )
