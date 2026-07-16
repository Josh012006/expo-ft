"""Config for EXPOLearnerOld: the original, reference-faithful ExpoFT
architecture (MSE scalar critic, REDQ-style ensemble) preserved unmodified in
expo_ft_old.py, for direct A/B comparison against the categorical
(XQC/XQCfD-style) critic rewrite now used by the default "EXPOLearner".

Reuses expo_ft_pi_config.py as-is — it already carries every field
expo_ft_old.create() needs (num_qs, num_min_qs, critic_layer_norm, etc.,
kept there specifically because SACLearner's architecture still needs them
too). The categorical-critic-specific fields also present there (num_atoms,
v_min, kl_coef, ...) are simply ignored by expo_ft_old.create(), whose
signature ends in **kwargs.
"""

from configs.model import expo_ft_pi_config


def get_config():
    config = expo_ft_pi_config.get_config()

    config.model_cls = "EXPOLearnerOld"

    return config
