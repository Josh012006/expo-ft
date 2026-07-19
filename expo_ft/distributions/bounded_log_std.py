"""Smooth, differentiable bounding for policy log-std parameters.

Standard fix for a known failure mode of jnp.clip(x, min, max): the
gradient of a hard clip is EXACTLY zero for any x at or beyond a bound, so
a parameter that starts at -- or drifts to -- a bound can never move away
from it again, no matter how strongly the loss would otherwise push it
there. This surfaced concretely in HetStat (hetstat.py), whose variance
head is deliberately initialized so its pre-bound log-std lands at
log_std_max (matching XQCfD's "near max entropy at init", Eq. 3/Fig. 1) --
with a hard clip, that init point is also a dead-gradient point, so the
variance head could never learn the paper's intended per-state variance
shaping.

This "soft clip" (two composed softplus saturations, one per side)
approaches the same [log_std_min, log_std_max] bound asymptotically but
keeps a live, non-vanishing gradient everywhere, including exactly at
either bound (gradient there is softplus'(0) = sigmoid(0) = 0.5, not 0).

Trade-off, by construction and unavoidable for any smooth bounded function:
the realized value at the old hard-clip boundary point is now slightly
inside it rather than exactly at it (e.g. with the default log_std_max=2,
an input of 2.0 now maps to ~1.31 instead of 2.0 -- still high/near-max
entropy in spirit, just not the literal asymptotic ceiling, in exchange for
a healthy gradient there instead of a dead one).

Purely a function of log_std_min/log_std_max -- no task-specific constants,
so it applies identically to every task and both distribution heads
(Normal/TanhNormal in tanh_normal.py and HetStatNormal/HetStatTanhNormal in
hetstat.py) that use it.
"""

import jax.nn as jnn


def smooth_log_std_bound(log_std_raw, log_std_min, log_std_max):
    """Smoothly bound log_std_raw into (log_std_min, log_std_max).

    Drop-in replacement for jnp.clip(log_std_raw, log_std_min, log_std_max)
    that keeps a non-vanishing gradient at and near both bounds.
    """
    bounded = log_std_max - jnn.softplus(log_std_max - log_std_raw)
    bounded = log_std_min + jnn.softplus(bounded - log_std_min)
    return bounded
