"""HetStat: heteroscedastic + stationary policy architecture.

Source: XQCfD (arXiv 2605.10734), Section 3.1 "Maximum Entropy Policies
using Stationary Features", Eq. 4 and surrounding text. Originally proposed
by Watson et al. 2023 ("Coherent soft imitation learning"), cited there.

THE PROBLEM THIS SOLVES
------------------------
A standard MLP policy head has UNDEFINED behavior out-of-distribution --
nothing constrains its predictions once the input state leaves the region
the network was actually trained on. This is a problem specifically for
KL regularization against a fixed/BC-derived reference (see expo_ft.py's
kl_coef): if the residual policy behaves arbitrarily (and possibly
confidently) in states the demonstrations never covered, the KL penalty
either fights against genuinely useful exploration there, or fails to
constrain a policy that is quietly making things up. XQCfD's fix: use a
policy architecture that reverts to a wide, close-to-uninformative
distribution specifically when out of distribution, so the KL term never
has to fight the network's own OOD behavior -- it can just do its job
in-distribution.

THE MECHANISM (paper Eq. 4 + surrounding text)
------------------------------------------------
Replace the network's usual final hidden layer with a FIXED (never
trained) random Fourier feature (RFF) projection of a stationary kernel:

    phi(s) = sqrt(2/K) * cos(V @ h(s) + b),   V_ij ~ N(0,1), b ~ U(0, 2*pi)

where h(s) is the ordinary MLP backbone's own output and V, b are sampled
ONCE from a fixed seed and never touched again -- not by gradients, not by
checkpointing (see below for why this needs no special handling). This is
the standard Rahimi & Recht (2007) construction for a stationary
(translation-invariant) kernel: because k(s,s) = k(0) is the SAME constant
for every s (that is literally what "stationary" means for a kernel), this
particular phi(s) has ||phi(s)||^2 approximately constant across every s.

On top of this fixed, stationary feature space, the paper places a
Bayesian-linear-style Gaussian: p(z|s) = N(mu^T phi(s), phi(s)^T Sigma
phi(s)), with mu and (diagonal) Sigma as the only LEARNED parameters. Since
phi(s)'s own norm barely varies with s, the aggregate predicted variance
stays roughly constant everywhere by default -- but Sigma can still be
shaped, per RFF *direction*, by training: directions of phi-space that get
consistently excited by in-distribution states get pushed toward LOW
variance specifically there, while directions never excited during
training keep their large initial value. This approximates "uncertainty
grows away from the training data" using an ordinary, backprop-trainable
parameter -- no exact closed-form Bayesian posterior update needed.

WHY V/b NEED NO SPECIAL HANDLING (no self.param, no self.variable)
--------------------------------------------------------------------
V and b are built from jax.random.PRNGKey(rff_seed) directly, as plain JAX
arrays -- NOT Flax params or variables. Since a fixed seed always produces
the exact same array, this reconstructs the identical V/b on every single
call (init, every training step, every checkpoint resume) with zero drift
and zero need to persist or restore them explicitly. This is deliberately
simpler than the alternative (storing them as a non-trainable Flax
variable collection, which would need every apply_fn call site AND the
checkpoint split/merge logic updated to thread it through) while being
exactly as "fixed forever" as the paper requires -- "V is a fixed random
weight matrix" (Eq. 4).

WHAT THIS FILE DOES NOT COVER
-------------------------------
The paper's "faithful" heteroscedastic BC loss (Eq. 5 -- MSE on the mean
head, NLLH-only with a stop-gradient on the variance head, both applied
during a policy-pretraining phase) is a property of a training LOSS, not
this architecture. This project does not currently have a residual-policy
BC-pretraining step (only critic pretraining exists so far -- see
update_critic's critic_pretrain_steps) -- there is no natural "residual
action" label in raw demonstrations to pretrain against directly. The
stop-gradient described above is still applied here, since it is a
property of how mean/variance interact architecturally regardless of which
loss trains them, but the MSE+NLLH split of Eq. 5 specifically is not
implemented, since there is nothing yet that would use it.
"""

import functools
from typing import Optional, Type

import jax
import jax.numpy as jnp
import flax.linen as nn
import tensorflow_probability.substrates.jax as tfp

from expo_ft.distributions.tanh_transformed import TanhTransformedDistribution
from expo_ft.networks import default_init

tfd = tfp.distributions


class HetStatNormal(nn.Module):
    """Drop-in replacement for expo_ft.distributions.Normal -- identical
    __call__ signature (base_cls(inputs, *args, **kwargs) -> features in,
    a squashable tfd.Distribution out), so it slots into
    residual_actor_cls in expo_ft.py's create() with no other code changes
    needed: sampling, log_prob, and the KL-regularization code (which reads
    dist.distribution.mean()/.stddev() -- see expo_ft.py's
    residual_actor_loss_fn) all keep working unmodified, since the output
    is still a TanhTransformedDistribution wrapping a
    tfd.MultivariateNormalDiag exactly like the original.
    """

    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    squash_tanh: bool = False
    pre_tanh_scale: float = 1.0
    num_rff_features: int = 256   # K -- dimensionality of the stationary feature space phi(s)
    rff_seed: int = 0             # fixes V, b forever -- see module docstring for why this needs no Flax param/variable

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        h = self.base_cls()(inputs, *args, **kwargs)  # ordinary MLP backbone -- same as Normal/TanhNormal
        hidden_dim = h.shape[-1]

        # ---- Fixed random Fourier feature (RFF) projection: never trained ----
        v_key, b_key = jax.random.split(jax.random.PRNGKey(self.rff_seed))
        V = jax.random.normal(v_key, (hidden_dim, self.num_rff_features))
        b = jax.random.uniform(b_key, (self.num_rff_features,), minval=0.0, maxval=2 * jnp.pi)
        # sqrt(2/K)*cos(...): standard Rahimi & Recht (2007) RFF construction
        # for a stationary kernel -- normalized so E[||phi(s)||^2] = k(s,s)
        # = k(0), the SAME constant for every s (Eq. 4).
        phi = jnp.sqrt(2.0 / self.num_rff_features) * jnp.cos(h @ V + b)

        # ---- Mean head: ordinary learned linear readout on the stationary features ----
        means = nn.Dense(self.action_dim, kernel_init=default_init(), name="OutputDenseMean")(phi)

        # ---- Variance head: quadratic form phi(s)^T Sigma phi(s), Sigma diagonal + learned ----
        # (paper's p(z|s) = N(mu^T phi(s), phi(s)^T Sigma phi(s))). Stop-
        # gradient on phi() feeding this head specifically -- the "faithful"
        # heteroscedastic-regression trick (Stirn et al. 2023, cited by
        # XQCfD): without it, the shared features tend to get pulled toward
        # whatever reduces the variance loss, under-fitting the mean.
        phi_for_var = jax.lax.stop_gradient(phi)
        # Init so the SUMMED initial variance lands near exp(2*log_std_max)
        # -- i.e. maximum allowed spread everywhere at init, matching the
        # paper's "uniform prior at initialization" (Eq. 3 / Figure 1).
        # NOTE: sum_k(phi_k(s)^2) ~= 1 in TOTAL (that's the whole point of
        # the sqrt(2/K) normalization -- see module docstring), not ~= 1
        # per individual feature. So with all K per-feature variances set
        # equal to a constant c, the summed variance is c * sum_k(phi_k^2)
        # ~= c * 1 = c directly -- no division by num_rff_features needed.
        init_var_per_feature = jnp.maximum(jnp.exp(2.0 * self.log_std_max), 1e-6)
        init_log_sigma_sq = jnp.log(jnp.expm1(init_var_per_feature))  # softplus^-1, so softplus(init) recovers init_var_per_feature
        log_sigma_sq = self.param(
            "HetStatLogVarScale",
            nn.initializers.constant(init_log_sigma_sq),
            (self.action_dim, self.num_rff_features),
            jnp.float32,
        )
        sigma_sq = jax.nn.softplus(log_sigma_sq)  # (action_dim, K), each >= 0 -- this IS diag(Sigma), per output dim
        variance = jnp.einsum("ik,...k->...i", sigma_sq, phi_for_var ** 2)  # phi^T diag(Sigma) phi, per action dim
        log_stds = 0.5 * jnp.log(variance + 1e-8)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        m = jnp.asarray(self.pre_tanh_scale, dtype=means.dtype)
        distribution = tfd.MultivariateNormalDiag(loc=means * m, scale_diag=jnp.exp(log_stds) * m)

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution


HetStatTanhNormal = functools.partial(HetStatNormal, squash_tanh=True)
