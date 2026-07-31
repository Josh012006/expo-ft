"""Config for EXPOLearnerCategorical: the categorical/distributional
(XQC/XQCfD-style, C51-style bounded support) critic rewrite, kept in
expo_ft_categorical.py for direct A/B comparison against the MSE
scalar-critic/REDQ-ensemble architecture now used by the default
"EXPOLearner" (see expo_ft.py). This used to be the default itself; it moved
here once the MSE architecture proved more stable under the corrected
sparse-reward setup (see the research report for the comparison).

Reuses expo_ft_pi_config.py as-is — it already carries every field
expo_ft_categorical.create() needs (num_atoms, v_min, v_max, kl_coef,
reward_scale_decay, use_reward_normalization, use_hetstat_policy, etc.). The
MSE/REDQ-specific fields also present there (num_qs, num_min_qs,
critic_layer_norm — kept there specifically because SACLearner's
architecture still needs them too) are simply ignored by
expo_ft_categorical.create(), whose signature ends in **kwargs.
"""

from configs.model import expo_ft_pi_config


def get_config():
    config = expo_ft_pi_config.get_config()

    config.model_cls = "EXPOLearnerCategorical"

    return config
