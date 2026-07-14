"""Categorical (distributional, C51-style) state-action value network and
supporting utilities, per XQC (arXiv 2509.25174) / XQCfD (arXiv 2605.10734).

Core idea: replace the old critic's unbounded scalar regression (a plain
MLP head trained with MSE against an unbounded TD target) with a
classification head over a FIXED, BOUNDED set of `num_atoms` values spanning
[v_min, v_max]. The critic outputs a probability distribution over this
support; Q = E[atoms] under that distribution. Structurally, Q can never
leave [v_min, v_max] — this is what directly targets the unbounded
target_q_max growth this whole project's debugging has been chasing,
independent of whatever remains true about the argmax-over-candidates
selection mechanism itself.

Also implements the other two XQC/XQCfD architectural pillars used
alongside the categorical head:
  - A Dense -> BatchNorm -> ReLU critic MLP (replacing the old plain
    MLP + optional LayerNorm), with BatchNorm applied to the network's
    input too.
  - Post-optimizer-step weight normalization: project each Dense layer's
    kernel columns to unit L2 norm after every gradient step.

No ensemble (XQCfD's own reported setup uses none) — a single critic +
single target critic, matching the "no Q-function ensembles" design choice
explicitly called out in the XQCfD abstract.
"""
from typing import Any, Sequence, Type

import flax.linen as nn
import jax
import jax.numpy as jnp


# ─────────────────────────── Network architecture ───────────────────────────

class _DenseBNReLUBlock(nn.Module):
    """One Dense -> BatchNorm -> ReLU block (XQC's critic MLP building block)."""
    features: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(self.features)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        return x


class XQCCriticBase(nn.Module):
    """4-layer Dense->BN->ReLU stack, 512-wide by default (XQC's spec).
    BatchNorm is applied to the network's INPUT as well, before the first
    Dense layer, per the paper."""
    hidden_dims: Sequence[int] = (512, 512, 512, 512)

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.BatchNorm(use_running_average=not train)(x)
        for dim in self.hidden_dims:
            x = _DenseBNReLUBlock(features=dim)(x, train=train)
        return x


class CategoricalStateActionValue(nn.Module):
    """(observations, actions) -> logits over `num_atoms` fixed support
    values. Q (scalar, for argmax/candidate-selection purposes) is NOT
    computed here — see q_from_logits() below; this module only produces
    the raw logits needed for both the cross-entropy loss and the
    expectation.

    Accepts (and ignores) a `sample_num` kwarg purely so it slots into
    PixelMultiplexer's calling convention unchanged (PixelMultiplexer always
    passes `sample_num`, a leftover assumption from when critics were always
    Ensemble-wrapped — see the PPO value-network `sample_num` bug this
    project hit and fixed for the exact same reason, in a different network).
    """
    base_cls: Any
    num_atoms: int = 101

    @nn.compact
    def __call__(self, observations, actions, train: bool = True, sample_num=None):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        x = self.base_cls()(inputs, train=train)
        logits = nn.Dense(self.num_atoms, name="AtomLogits")(x)
        return logits


def make_atoms(num_atoms: int, v_min: float, v_max: float) -> jnp.ndarray:
    """The fixed, shared support atoms — identical for the online critic and
    the target critic; only the PROBABILITY MASS placed on each atom differs
    between them. Never trained, never moves."""
    return jnp.linspace(v_min, v_max, num_atoms)


def q_from_logits(logits: jnp.ndarray, atoms: jnp.ndarray) -> jnp.ndarray:
    """Q = E_{a ~ softmax(logits)}[atom_value(a)]. Structurally bounded to
    [atoms.min(), atoms.max()] = [v_min, v_max] no matter what the logits
    are — this is the actual mechanism preventing unbounded Q growth."""
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(probs * atoms, axis=-1)


# ─────────────────────────── Categorical Bellman projection ───────────────────

def categorical_bellman_projection(
    next_probs: jnp.ndarray,   # (batch, num_atoms) -- softmax(next_logits)
    rewards: jnp.ndarray,      # (batch,)
    masks: jnp.ndarray,        # (batch,) -- 1 - terminated
    discount: float,
    atoms: jnp.ndarray,        # (num_atoms,)
    v_min: float,
    v_max: float,
) -> jnp.ndarray:
    """The standard C51 categorical Bellman projection (Bellemare, Dabney &
    Munos, 2017, "A Distributional Perspective on Reinforcement Learning").

    The Bellman-shifted return distribution (reward + discount * mask * atoms)
    generally does NOT land back on the fixed support atoms -- e.g. a shifted
    atom might fall between two of the original atom positions. Standard C51
    projection: clip the shifted atoms into [v_min, v_max], then distribute
    each shifted atom's probability mass onto its two nearest neighboring
    fixed atoms, proportionally to distance (linear interpolation), so the
    result is a valid probability distribution over the SAME fixed atoms the
    critic itself predicts over. This projected distribution is the
    cross-entropy target for the critic loss.
    """
    num_atoms = atoms.shape[0]
    delta_z = (v_max - v_min) / (num_atoms - 1)

    # Shifted, clipped atom locations for each batch element: (batch, num_atoms)
    tz = rewards[:, None] + discount * masks[:, None] * atoms[None, :]
    tz = jnp.clip(tz, v_min, v_max)

    # Fractional index of each shifted atom into the fixed grid.
    b = (tz - v_min) / delta_z
    lower = jnp.floor(b)
    upper = jnp.ceil(b)

    # Distribute mass: where b lands exactly on a fixed atom (lower==upper),
    # all mass goes there; otherwise split proportionally to distance.
    lower_weight = (upper - b) + jnp.where(lower == upper, 1.0, 0.0)
    upper_weight = (b - lower)

    lower_idx = lower.astype(jnp.int32)
    upper_idx = upper.astype(jnp.int32)

    batch_size = next_probs.shape[0]
    target_probs = jnp.zeros((batch_size, num_atoms))

    batch_idx = jnp.arange(batch_size)[:, None]
    batch_idx = jnp.broadcast_to(batch_idx, (batch_size, num_atoms))

    target_probs = target_probs.at[batch_idx, lower_idx].add(next_probs * lower_weight)
    target_probs = target_probs.at[batch_idx, upper_idx].add(next_probs * upper_weight)

    return target_probs


def categorical_cross_entropy_loss(logits: jnp.ndarray, target_probs: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy between the critic's predicted distribution
    (softmax(logits)) and the projected Bellman target distribution.
    `target_probs` is a fixed (stop-gradient'd by construction, since it's
    built from the target critic + rewards) soft label, not a hard class."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(target_probs * log_probs, axis=-1)


# ─────────────────────────── Weight normalization (post-step) ───────────────

def project_weights_to_unit_norm(params):
    """XQC's weight normalization: after each optimizer step, project every
    Dense layer's kernel so each OUTPUT unit's incoming weight vector
    (i.e. each column of the kernel matrix) has unit L2 norm. Applied as a
    pure post-processing step on the params pytree — not a reparametrized
    layer, not part of the gradient computation itself, just called once
    right after apply_gradients().

    Only touches leaves named "kernel" with ndim==2 (standard nn.Dense
    kernels); everything else (biases, BatchNorm scale/bias/mean/var,
    higher-rank arrays) passes through unchanged.
    """
    def _project(path, leaf):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if name == "kernel" and leaf.ndim == 2:
            col_norms = jnp.linalg.norm(leaf, axis=0, keepdims=True)
            return leaf / jnp.maximum(col_norms, 1e-8)
        return leaf

    return jax.tree_util.tree_map_with_path(_project, params)
