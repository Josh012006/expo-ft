"""EXPO-FT learner: Pi0.5 base policy + residual actor + critic (EXPOLearner).

Critic architecture: categorical/distributional (C51-style, bounded support)
per XQC (arXiv 2509.25174) / XQCfD (arXiv 2605.10734), replacing the earlier
scalar-regression critic (still available, unmodified, in expo_ft_old.py, for
comparison/rollback). See expo_ft/networks/categorical_value.py for the
network, Bellman projection, and weight-normalization implementation. No
critic ensemble (XQCfD's own reported setup uses none) — a single critic and
single target critic.

Everything ELSE (the residual/edit policy, the argmax-over-candidates
action-selection mechanism itself, the base VLA fine-tuning, checkpointing,
replay/data plumbing) is unchanged from expo_ft_old.py.
"""

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import dataclasses

import logging

import flax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import struct
from flax.training.train_state import TrainState

import numpy as np

from expo_ft.agents.alg.agent import AgentLearner, initialize_checkpoint_dir
from expo_ft.agents.alg.checkpoint_utils import make_checkpoint_fns
from expo_ft.agents.alg.batch_utils import prepare_critic_batch, prepare_actor_sampling_batch, extract_critic_fields
from expo_ft.networks.temperature import Temperature
from expo_ft.data.dataset import DatasetDict
from expo_ft.distributions import TanhNormal, HetStatTanhNormal
from expo_ft.networks import (
    MLP,
    PixelMultiplexer,
    PixelEditMultiplexer,
    BatchEncoder,
)
from expo_ft.networks.encoders import ResNetV2Encoder
from expo_ft.networks.categorical_value import (
    XQCCriticBase,
    CategoricalStateActionValue,
    make_atoms,
    q_from_logits,
    categorical_bellman_projection,
    categorical_cross_entropy_loss,
    project_weights_to_unit_norm,
)

from expo_ft.utils.augmentation import make_data_augmentation_fn

import openpi.shared.array_typing as at
import openpi.training.sharding as _sharding
import openpi.training.utils as training_utils


class BNTrainState(TrainState):
    """TrainState extended with a batch_stats field, for the critic's
    BatchNorm running mean/var — a separate mutable collection from `params`
    that flax's plain TrainState doesn't track. Updated via
    .replace(batch_stats=...) after each apply() call with mutable=
    ['batch_stats'], NOT via apply_gradients (batch_stats isn't
    gradient-updated, it's a running average maintained by BatchNorm itself
    during the forward pass)."""
    batch_stats: Any = None


def _split_params(agent: Any) -> tuple[Any, dict[str, at.Params]]:
    batch_encoder_params = agent.batch_encoder.params
    residual_actor_params = agent.residual_actor.params
    temp_params = agent.temp.params
    critic_params = agent.critic.params
    critic_batch_stats = agent.critic.batch_stats

    with at.disable_typechecking():
        if agent.actor_train_state.ema_params is not None:
            actor_params = agent.actor_train_state.ema_params
            actor_train_state = dataclasses.replace(agent.actor_train_state, ema_params=None)
        else:
            actor_params = agent.actor_train_state.params
            actor_train_state = dataclasses.replace(agent.actor_train_state, params={})

    agent = dataclasses.replace(
        agent, 
        batch_encoder=dataclasses.replace(agent.batch_encoder, params={}),
        residual_actor=dataclasses.replace(agent.residual_actor, params={}), 
        temp=dataclasses.replace(agent.temp, params={}), 
        critic=dataclasses.replace(agent.critic, params={}, batch_stats={}),
        actor_train_state=actor_train_state
    )

    params = {
        "batch_encoder_params": batch_encoder_params,
        "residual_actor_params": residual_actor_params, 
        "temp_params": temp_params, 
        "critic_params": critic_params,
        "critic_batch_stats": critic_batch_stats,
        "actor_params": actor_params
    }
    return agent, params


def _merge_params(agent: Any, params: dict[str, at.Params]) -> Any:
    batch_encoder = dataclasses.replace(agent.batch_encoder, params=params["batch_encoder_params"])
    residual_actor = dataclasses.replace(agent.residual_actor, params=params["residual_actor_params"])
    temp = dataclasses.replace(agent.temp, params=params["temp_params"])
    critic = dataclasses.replace(agent.critic, params=params["critic_params"], batch_stats=params["critic_batch_stats"])

    with at.disable_typechecking():
        if agent.actor_train_state.params:
            actor_train_state = dataclasses.replace(agent.actor_train_state, ema_params=params["actor_params"])
        else:
            actor_train_state = dataclasses.replace(agent.actor_train_state, params=params["actor_params"])

    agent = dataclasses.replace(
        agent, 
        batch_encoder=batch_encoder,
        residual_actor=residual_actor,
        temp=temp,
        critic=critic,
        actor_train_state=actor_train_state,
    )
    return agent


_restore_checkpoint, _save_checkpoint = make_checkpoint_fns(_split_params, _merge_params)


def restore_checkpoint(checkpoint_manager, agent, step: int | None = None):
    return _restore_checkpoint(checkpoint_manager, agent, step)

def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    agent: Any,
    step: int,
):
    _save_checkpoint(checkpoint_manager, agent, step)

def load_agent(seed, example_observation, example_action, example_state,
               actor, actor_train_state, target_actor_params, agent_kwargs, metadata,
               mesh, data_sharding, replicated_sharding, resume, replan_steps,
               default_prompt, residual_action_xyzg):
    """Create an EXPOLearner from a pre-built VLA actor and remaining config kwargs."""
    agent_kwargs.update(
        actor=actor,
        actor_train_state=actor_train_state,
        target_actor_params=target_actor_params,
        mesh=mesh,
        resume=resume,
        replan_steps=replan_steps,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        default_prompt=default_prompt,
        residual_action_xyzg=residual_action_xyzg,
        **metadata,
    )
    return EXPOLearner.create(seed, example_observation, example_action, example_state, **agent_kwargs)

def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    # NOTE: previously wrapped in flax.core.FrozenDict(...) here, which crashes
    # optax.masked's internal jax.tree.map — actual params (critic_params,
    # batch_encoder_params) are plain dicts in this flax version, not
    # FrozenDict, so mask and params pytree structures didn't match
    # ("Custom node type mismatch"). Latent bug, never triggered before
    # since critic_weight_decay had never actually been set to a non-null
    # value in any prior run.
    return flax.traverse_util.unflatten_dict(flat_mask)

@partial(jax.jit, static_argnames=('critic_fn',))
def compute_q(critic_fn, critic_params, critic_batch_stats, atoms, observations, actions, states):
    """Q = E[atoms] under the critic's predicted categorical distribution —
    see expo_ft/networks/categorical_value.py. Structurally bounded to
    [atoms.min(), atoms.max()] no matter what the network outputs.

    Always evaluated in inference mode (training=False, i.e. BatchNorm uses
    its running mean/var rather than this batch's) — this function is used
    for action SELECTION (scoring the target critic's candidates), never for
    the critic's own training step (see update_critic's own logits/loss
    computation for that, which correctly uses training=True instead).

    No ensemble reduction anymore (no more `.min(axis=0)` over a REDQ-style
    ensemble) — single critic, single target critic, per XQCfD's own
    reported setup ("no Q-function ensembles").
    """
    logits = critic_fn(
        {"params": critic_params, "batch_stats": critic_batch_stats},
        observations, actions, False, p=states,
    )
    return q_from_logits(logits, atoms)


@partial(jax.jit, static_argnames=('encoder_fn', 'stop_gradient'))
def batch_encode(encoder_fn, encoder_params, observations, stop_gradient=False):
    encoded = encoder_fn({'params': encoder_params}, observations, stop_gradient=stop_gradient)
    return encoded


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: jnp.ndarray, states, actions) -> jnp.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations, actions=actions, p=states)
    return dist.sample(seed=key), rng


class EXPOLearner(AgentLearner, struct.PyTreeNode):
    rng: jax.random.PRNGKey
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    critic: TrainState
    batch_encoder: TrainState
    target_critic: TrainState
    actor: Any = struct.field(pytree_node=False)
    actor_train_state: training_utils.TrainState
    target_actor_params: at.Params
    residual_actor: TrainState
    temp: TrainState
    N: int = struct.field(pytree_node=False)
    n_edit_samples: int = struct.field(pytree_node=False)
    edit_scale: float = struct.field(pytree_node=False)
    residual_action_xyzg: bool = struct.field(pytree_node=False)
    batch_split: int = struct.field(pytree_node=False)
    encode_batch_split: int = struct.field(pytree_node=False)
    actor_tau: float
    tau: float
    discount: float
    target_entropy: float
    entropy_scale: float
    num_atoms: int = struct.field(pytree_node=False)
    v_min: float = struct.field(pytree_node=False)
    v_max: float = struct.field(pytree_node=False)
    atoms: jnp.ndarray  # fixed support values, e.g. linspace(v_min, v_max, num_atoms) — a pytree leaf (JAX arrays can't be static/hashable fields), but never trained, never changes after create()
    reward_scale_decay: float = struct.field(pytree_node=False)
    kl_coef: float = struct.field(pytree_node=False)  # XQCfD-style KL-to-reference regularization for the edit policy; 0.0 = disabled (default), matches pre-existing entropy-only behavior
    kl_ref_std: float = struct.field(pytree_node=False)  # std of the fixed N(0, kl_ref_std) reference in pre-tanh space
    reward_ms: jnp.ndarray  # running mean-square of rewards (EMA), used to normalize rewards before the Bellman projection so the FIXED [v_min, v_max] support stays meaningful regardless of a task's absolute reward scale — see update_critic()
    action_dim: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    full_action_dim: int = struct.field(pytree_node=False)
    replan_steps: int = struct.field(pytree_node=False)
    action_horizon: int = struct.field(pytree_node=False)
    resize_size: Optional[int] = struct.field(pytree_node=False)
    default_prompt: Optional[str] = struct.field(pytree_node=False)
    data_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    batch_encoder_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    freeze_encoder: Optional[bool] = struct.field(pytree_node=False)
    freeze_critic_encoder: bool = struct.field(pytree_node=False)
    actor_success_only: bool = struct.field(pytree_node=False)
    fixed_temperature: Optional[float] = struct.field(pytree_node=False, default=None)
    critic_grad_clip_norm: Optional[float] = struct.field(pytree_node=False, default=None)
    _infer_cache: Optional[dict] = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        states,
        # Pre-built VLA actor (constructed by build_pi05 or similar factory)
        actor: Any = None,
        actor_train_state: Any = None,
        target_actor_params: Any = None,
        # VLA metadata (extracted by factory from backbone config)
        action_horizon: int = 1,
        mesh: Optional[Any] = None,
        freeze_encoder: bool = False,
        # EXPO-specific params
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        # Categorical (C51-style, bounded support) critic — replaces the old
        # ensemble-of-scalars critic (num_qs/num_min_qs/critic_layer_norm/
        # critic_dropout_rate/use_critic_resnet are gone; see
        # expo_ft/networks/categorical_value.py).
        #
        # v_min/v_max apply to NORMALIZED reward units, not raw task reward
        # scale — rewards are divided by a running RMS estimate
        # (reward_scale_decay controls the EMA) before the Bellman
        # projection, so this fixed support stays meaningful across any
        # task's absolute reward scale without per-task hand-tuning (the
        # atoms themselves never move/reproject — only the reward's
        # normalization does, avoiding any instability from redefining a
        # trained distributional head's support mid-training). Still watch
        # target_q_max/min: if Q sits pinned at exactly v_min or v_max for a
        # meaningful fraction of training even in normalized units, the
        # support itself (not just the normalization) is too narrow.
        num_atoms: int = 101,
        v_min: float = -10.0,
        v_max: float = 20.0,
        reward_scale_decay: float = 0.99,
        # XQCfD-style KL regularization for the edit/residual policy,
        # replacing (when enabled) the generic entropy bonus with a penalty
        # for deviating from a fixed N(0, kl_ref_std) reference in pre-tanh
        # space — "prefer staying close to zero residual unless Q strongly
        # justifies deviating". 0.0 = disabled (exact no-op), matching
        # pre-existing behavior.
        kl_coef: float = 0.0,
        kl_ref_std: float = 1.0,
        # HetStat (heteroscedastic + stationary) residual-policy architecture,
        # per XQCfD Section 3.1 -- see expo_ft/distributions/hetstat.py for
        # the full mechanism. Replaces TanhNormal's usual MLP-head-directly
        # architecture with one that reverts to a wide, near-uniform
        # distribution when out of the demos' distribution, so kl_coef's
        # regularization doesn't fight the network's own OOD behavior.
        # False = disabled (default; matches pre-existing TanhNormal
        # behavior exactly, no change unless explicitly turned on).
        use_hetstat_policy: bool = False,
        hetstat_num_rff_features: int = 256,
        critic_hidden_dims: Sequence[int] = (512, 512, 512, 512),
        critic_weight_decay: Optional[float] = None,
        critic_grad_clip_norm: Optional[float] = None,
        target_entropy: Optional[float] = None,
        adjust_target_entropy: Optional[bool] = False,
        entropy_scale: float = 1.0,
        init_temperature: float = 1.0,
        fixed_temperature: Optional[float] = None,
        use_pnorm: bool = False,
        actor_drop: Optional[float] = None,
        N: int = 32,
        batch_split: int = 1,
        encode_batch_split: int = 1,
        n_edit_samples: int = 0,
        edit_scale: float = 1.0,
        residual_action_xyzg: bool = False,
        actor_tau: float = 0.001,
        include_state: bool = True,
        latent_dim_image: int = 50,
        latent_dim_state: int = 50,
        encoder_stage_sizes: Tuple[int, int, int, int] = (2, 2, 2, 2),
        encoder_num_filters: int = 64,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
        resume: bool = False,
        replan_steps: int = 1,
        freeze_critic_encoder: bool = False,
        data_sharding: Optional[jax.sharding.NamedSharding] = None,
        replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
        default_prompt: Optional[str] = None,
        resize_size: Optional[int] = None,
        actor_success_only: bool = False,
        use_full_augmentation: bool = True,
        **kwargs,
    ):
        action_dim = action_space.shape[-1]
        state_dim = states.shape[-1]
        full_action_dim = replan_steps * action_dim
        observations = observation_space
        actions = jnp.zeros((full_action_dim,))
        print("observation shape: ", observations.shape)
        print("action shape: ", actions.shape, "action horizon: ", action_horizon, "action dim: ", action_dim)
        print("residual actor output size / q function input size: ", full_action_dim, " (replan_steps=", replan_steps, "* action_dim=", action_dim, ")")
        print("states shape: ", states.shape)

        if target_entropy is None:
            if adjust_target_entropy:
                target_entropy = -full_action_dim / 2 + full_action_dim * jnp.log(edit_scale)
            else:
                target_entropy = -full_action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        rng, encoder_key = jax.random.split(rng, 2)

        encoder_cls = partial(
            ResNetV2Encoder,
            stage_sizes=encoder_stage_sizes,
            num_filters=encoder_num_filters,
        )

        batch_encoder_def = BatchEncoder(
            encoder_cls=encoder_cls,
            latent_dim=latent_dim_image,
            pixel_keys=pixel_keys,
            depth_keys=depth_keys,
        )

        batch_encoder_params = batch_encoder_def.init(encoder_key, observations)["params"]
        # Same weight decay + clip norm as the critic head — the encoder's
        # gradients come from the same critic loss, so leaving it out here
        # would make this test regularize only half of what actually feeds
        # the exploding Q (same reasoning as the clip norm above).
        if critic_weight_decay is not None:
            encoder_tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            encoder_tx = optax.adam(learning_rate=critic_lr)
        if critic_grad_clip_norm is not None:
            encoder_tx = optax.chain(
                optax.clip_by_global_norm(critic_grad_clip_norm),
                encoder_tx,
            )
        batch_encoder = TrainState.create(
            apply_fn=batch_encoder_def.apply,
            params=batch_encoder_params,
            tx=encoder_tx,
        )

        batch_encoder_shape = jax.eval_shape(lambda: batch_encoder)
        batch_encoder_sharding = _sharding.fsdp_sharding(batch_encoder_shape, mesh, log=True)
        batch_encoder = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=batch_encoder_sharding,
        )(batch_encoder)

        critic_observations = jnp.ones((1, latent_dim_image))
        critic_actions = jnp.expand_dims(actions, axis = 0)
        critic_states = jnp.expand_dims(states, axis = 0)
        critic_states_ext = critic_states
        print("critic actions shape: ", critic_actions.shape)
        print("critic states shape: ", critic_states.shape)

        residual_actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm
        )
        residual_actor_cls = (
            HetStatTanhNormal(residual_actor_base_cls, full_action_dim, num_rff_features=hetstat_num_rff_features)
            if use_hetstat_policy
            else TanhNormal(residual_actor_base_cls, full_action_dim)
        )
        residual_actor_def = PixelEditMultiplexer(
            network_cls=residual_actor_cls,
            latent_dim=latent_dim_state,
            include_state=include_state,
        )

        residual_actor_params = residual_actor_def.init(actor_key, jnp.ones((1, latent_dim_image)), actions=jnp.ones((1, full_action_dim)), p=critic_states)["params"]
        residual_actor = TrainState.create(
            apply_fn=residual_actor_def.apply, 
            params=residual_actor_params, 
            tx=optax.adam(learning_rate=actor_lr),
        )

        residual_actor_shape = jax.eval_shape(lambda: residual_actor)
        residual_actor_sharding = _sharding.fsdp_sharding(residual_actor_shape, mesh, log=True)
        residual_actor = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=residual_actor_sharding,
        )(residual_actor)

        critic_base_cls = partial(
            XQCCriticBase,
            hidden_dims=critic_hidden_dims,
        )

        critic_cls = partial(CategoricalStateActionValue, base_cls=critic_base_cls, num_atoms=num_atoms)
        critic_def = PixelMultiplexer(
            network_cls=critic_cls,
            latent_dim=latent_dim_state,
            include_state=include_state,
        )
        # `True` for `training` here (not the old code's implicit default
        # False) so BatchNorm's `batch_stats` collection is actually created
        # at init time — .init() would still create it either way (flax
        # walks every declared variable collection during init regardless of
        # the train/eval flag value), but passing True is the more correct,
        # unambiguous signal of intent given this is about to be trained.
        critic_variables = critic_def.init(critic_key, critic_observations, critic_actions, True, p=critic_states_ext)
        critic_params = critic_variables["params"]
        critic_batch_stats = critic_variables["batch_stats"]
        atoms = make_atoms(num_atoms, v_min, v_max)
        reward_ms = jnp.array(1.0)  # start at scale=1.0 (no normalization effect) until enough real data accumulates

        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)

        # Gradient clipping on the critic (and its encoder, since the encoder's
        # gradients flow through the same update when freeze_critic_encoder=False).
        # Applied before the adam/adamw step transform, per standard practice —
        # clip the raw gradient direction/magnitude first, then let adam adapt
        # per-parameter scaling on top of the clipped gradient.
        if critic_grad_clip_norm is not None:
            tx = optax.chain(
                optax.clip_by_global_norm(critic_grad_clip_norm),
                tx,
            )

        critic = BNTrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            batch_stats=critic_batch_stats,
            tx=tx,
        )

        critic_shape = jax.eval_shape(lambda: critic)
        critic_sharding = _sharding.fsdp_sharding(critic_shape, mesh, log=True)
        critic = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=critic_sharding,
        )(critic)

        target_critic = BNTrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            batch_stats=critic_batch_stats,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        temp_shape = jax.eval_shape(lambda: temp)
        temp_sharding = _sharding.fsdp_sharding(temp_shape, mesh, log=True)
        temp = jax.jit(
            lambda x: x,
            in_shardings=replicated_sharding,
            out_shardings=temp_sharding,
        )(temp)


        agent = cls(
            rng=rng,
            actor=actor,
            actor_train_state=actor_train_state,
            target_actor_params=target_actor_params,
            residual_actor=residual_actor,
            N=N,
            n_edit_samples=n_edit_samples,
            encode_batch_split=encode_batch_split,
            edit_scale=edit_scale,
            residual_action_xyzg=residual_action_xyzg,
            batch_split=batch_split,
            actor_tau=actor_tau,
            critic=critic,
            target_critic=target_critic,
            batch_encoder=batch_encoder,
            temp=temp,
            target_entropy=target_entropy,
            entropy_scale=entropy_scale,
            tau=tau,
            discount=discount,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            atoms=atoms,
            reward_scale_decay=reward_scale_decay,
            kl_coef=kl_coef,
            kl_ref_std=kl_ref_std,
            reward_ms=reward_ms,
            fixed_temperature=fixed_temperature,
            critic_grad_clip_norm=critic_grad_clip_norm,
            data_augmentation_fn=make_data_augmentation_fn(use_full_augmentation),

            action_dim=action_dim,
            state_dim=state_dim,
            full_action_dim=full_action_dim,
            action_horizon=action_horizon,
            replan_steps=replan_steps,
            resize_size=resize_size,
            default_prompt=default_prompt,
            data_sharding=data_sharding,
            batch_encoder_sharding=batch_encoder_sharding,
            freeze_encoder=freeze_encoder,
            freeze_critic_encoder=freeze_critic_encoder,
            actor_success_only=actor_success_only,
        )
        if not resume:
            agent = agent.cache_infer_params()
        return agent

    def _apply_residual_xyzg_mask(self, residual: jnp.ndarray) -> jnp.ndarray:
        """Zero out rotation dims (3,4,5) when residual_action_xyzg is True. Keeps xyz (0,1,2) and gripper (6)."""
        if not self.residual_action_xyzg:
            return residual
        # mask: 1 for xyz (0,1,2) and gripper (last dim), 0 for rotation (3,4,5)
        mask = jnp.ones(self.action_dim).at[3:6].set(0.0)
        full_mask = jnp.tile(mask, self.replan_steps)
        return residual * full_mask

    def cache_infer_params(self):
        """Copy params onto infer_sharding for rollout sampling.

        sample_actions reads _infer_cache to avoid device_put on every env step.
        Call again after update() so rollouts use the latest weights.
        """
        s = self.actor.infer_sharding
        return self.replace(_infer_cache={
            "actor_train_state": jax.device_put(self.actor_train_state, s),
            "batch_encoder_params": jax.device_put(self.batch_encoder.params, s),
            "residual_actor_params": jax.device_put(self.residual_actor.params, s),
            "target_critic_params": jax.device_put(self.target_critic.params, s),
            "target_critic_batch_stats": jax.device_put(self.target_critic.batch_stats, s),
        })

    def _sample_residual(self, key, residual_params, encoded_obs, states, base_actions):
        r_samples, rng = _sample_actions(
            key, self.residual_actor.apply_fn, residual_params, encoded_obs, states, base_actions
        )
        residual_scaled = self._apply_residual_xyzg_mask(r_samples * self.edit_scale)
        combined = residual_scaled + base_actions
        return combined, residual_scaled, rng

    def _encode_observations(self, observations, encoder_params=None, stop_gradient=True):
        params = encoder_params or self.batch_encoder.params
        if self.encode_batch_split > 1:
            one_call = observations.shape[0] // self.encode_batch_split
            chunks = [
                batch_encode(self.batch_encoder.apply_fn, params, observations[i * one_call:(i + 1) * one_call], stop_gradient=stop_gradient)
                for i in range(self.encode_batch_split)
            ]
            return jnp.concatenate(chunks)
        return batch_encode(self.batch_encoder.apply_fn, params, observations, stop_gradient=stop_gradient)

    def _compute_q_split(self, critic_fn, critic_params, critic_batch_stats, obs, actions, states):
        if self.batch_split > 1:
            total = obs.shape[0]
            one_call = total // self.batch_split
            q_list = [
                compute_q(critic_fn, critic_params, critic_batch_stats, self.atoms, obs[i * one_call:(i + 1) * one_call], actions[i * one_call:(i + 1) * one_call], states[i * one_call:(i + 1) * one_call])
                for i in range(self.batch_split)
            ]
            return jnp.concatenate(q_list)
        return compute_q(critic_fn, critic_params, critic_batch_stats, self.atoms, obs, actions, states)

    def sample_actions(self, observations, only_base_actions=False):
        # Keep inference-time randomness on the same single device as inference params
        # to avoid cross-device gather/indexing mismatches.
        infer_sharding = self.actor.infer_sharding
        rng = jax.device_put(self.rng, infer_sharding)
        c = self._infer_cache or {}
        _actor_train_state = c.get("actor_train_state") or self.actor_train_state
        _batch_encoder_params = c.get("batch_encoder_params") or jax.device_put(self.batch_encoder.params, infer_sharding)
        _residual_actor_params = c.get("residual_actor_params") or jax.device_put(self.residual_actor.params, infer_sharding)
        _target_critic_params = c.get("target_critic_params") or jax.device_put(self.target_critic.params, infer_sharding)
        _target_critic_batch_stats = c.get("target_critic_batch_stats") or jax.device_put(self.target_critic.batch_stats, infer_sharding)

        transformed_inputs = self.actor.process_raw_inputs(observations, self.action_dim, self.resize_size)
        transformed_inputs = extract_critic_fields(transformed_inputs, self.actor.model_config.action_dim, self.state_dim)

        key, rng = jax.random.split(rng)
        transformed_actions, sample_time = self.actor.sample_actions(
            transformed_inputs,
            train_state=_actor_train_state,
            rng=key,
            train=False,
            num_samples=self.N,
        )
        raw_actions = self.actor.process_transformed_outputs(transformed_actions)
        
        if only_base_actions:
            action = raw_actions[0].reshape(self.action_horizon, self.action_dim)
            sample_info = {"sample_time": sample_time, "selected_action_type": "main"}
            return jnp.array(action), self.replace(rng=rng), sample_info

        transformed_full = transformed_actions  # (N, action_horizon, action_dim)
        transformed_actions = transformed_full[:, :self.replan_steps, :].reshape(self.N, self.full_action_dim)

        critic_transformed_obs = transformed_inputs["critic_obs"]
        transformed_states = transformed_inputs["critic_states"]
        
        if self.freeze_encoder:
            critic_transformed_obs = critic_transformed_obs.repeat(self.N, axis=0)
            transformed_states = transformed_states.repeat(self.N, axis=0)

        critic_encoded_obs = batch_encode(self.batch_encoder.apply_fn, _batch_encoder_params, critic_transformed_obs[:1], stop_gradient=True)
        critic_encoded_obs = critic_encoded_obs.repeat(self.N, axis=0)

        if self.N > 1:
            if self.n_edit_samples > 0:
                key, rng = jax.random.split(rng, 2)

                critic_encoded_obs = jnp.concatenate([critic_encoded_obs, jnp.expand_dims(critic_encoded_obs[0], axis = 0).repeat(self.n_edit_samples, axis = 0)], axis=0)
                transformed_states = jnp.concatenate([transformed_states, jnp.expand_dims(transformed_states[0], axis = 0).repeat(self.n_edit_samples, axis = 0)], axis=0)

                r_observations = jnp.repeat(jnp.expand_dims(critic_encoded_obs[0], axis = 0), self.n_edit_samples, axis=0)
                r_states = jnp.repeat(jnp.expand_dims(transformed_states[0], axis = 0), self.n_edit_samples, axis=0)
                base_actions = transformed_actions.copy()[:self.n_edit_samples]

                r_samples, _, rng = self._sample_residual(key, _residual_actor_params, r_observations, r_states, base_actions)
                transformed_actions = jnp.concatenate([base_actions, r_samples], axis=0)

                r_modified = r_samples.reshape(self.n_edit_samples, self.replan_steps, self.action_dim)
                full_r_modified = transformed_full[:self.n_edit_samples].at[:, :self.replan_steps, :].set(r_modified)
                raw_r_samples = self.actor.process_transformed_outputs(full_r_modified)
                raw_actions = jnp.concatenate([raw_actions, raw_r_samples], axis=0)

            qs = compute_q(self.target_critic.apply_fn, _target_critic_params, _target_critic_batch_stats, self.atoms, critic_encoded_obs, transformed_actions, transformed_states)

            idx = jnp.argmax(qs)

            action = raw_actions[idx]
        else:
            action = raw_actions[0]
            idx = 0

        action = action.reshape(self.action_horizon, self.action_dim)
        rng, _ = jax.random.split(rng, 2)
        sample_info = {"sample_time": sample_time}
        return jnp.array(action), self.replace(rng=rng), sample_info

    def sample_batch_actions(self, batch):
        critic_obs = jnp.squeeze(batch["next_observations"])
        critic_obs = jax.device_put(critic_obs)
        batch_size = critic_obs.shape[0]
        states = jnp.squeeze(batch["next_states"])
        states = jax.device_put(states, self.data_sharding)
        rng = self.rng

        # Prepare VLA inputs
        transformed_inputs = prepare_actor_sampling_batch(batch)
        if not self.freeze_encoder:
            def repeat_value(v, n):
                if v is None:
                    return None
                elif isinstance(v, dict):
                    return {k: repeat_value(vv, n) for k, vv in v.items()}
                else:
                    return jnp.repeat(v, n, axis=0)
            transformed_inputs = {k: repeat_value(v, self.N) for k, v in transformed_inputs.items()}

        # Encode observations
        encoded_obs = self._encode_observations(critic_obs, stop_gradient=True)
        encoded_obs = jax.device_put(encoded_obs, self.data_sharding)

        # Sample base actions from VLA
        key, rng = jax.random.split(rng)
        actor_actions, sample_time = self.actor.sample_training_actions(
            transformed_inputs=transformed_inputs,
            train_state=self.actor_train_state,
            rng=key,
            train=False,
            num_samples=self.N if self.freeze_encoder else 1,
        )
        actor_actions = actor_actions[:, :self.replan_steps, :]
        actions = actor_actions.reshape(batch_size, self.N, self.full_action_dim)

        # Sample residual actions
        total_candidates = self.N + self.n_edit_samples

        if self.n_edit_samples > 0:
            key, rng = jax.random.split(rng, 2)
            r_observations = jax.device_put(jnp.repeat(encoded_obs, self.n_edit_samples, axis=0), self.data_sharding)
            r_states = jax.device_put(jnp.repeat(states, self.n_edit_samples, axis=0), self.data_sharding)
            d_actions = actions.copy()[:, :self.n_edit_samples].reshape(-1, actions.shape[-1])

            r_samples, residual_scaled, rng = self._sample_residual(key, self.residual_actor.params, r_observations, r_states, d_actions)
            mean_d_actions_norm = jnp.mean(jnp.linalg.norm(d_actions, axis=1))
            mean_residual_scaled_norm = jnp.mean(jnp.linalg.norm(residual_scaled, axis=1))

            actions = jnp.concatenate([actions, r_samples.reshape(batch_size, self.n_edit_samples, -1)], axis=1)
        
        else:
            mean_d_actions_norm = jnp.array(0.0, dtype=jnp.float32)
            mean_residual_scaled_norm = jnp.array(0.0, dtype=jnp.float32)

        # Select best actions via Q-values
        if self.N > 1:
            obs_flat = jax.device_put(jnp.repeat(encoded_obs, total_candidates, axis=0), self.data_sharding)
            states_flat = jax.device_put(jnp.repeat(states, total_candidates, axis=0), self.data_sharding)
            actions_flat = jax.device_put(actions.reshape(-1, actions.shape[-1]), self.data_sharding)

            qs = self._compute_q_split(self.target_critic.apply_fn, self.target_critic.params, self.target_critic.batch_stats, obs_flat, actions_flat, states_flat)
            qs = qs.reshape(batch_size, total_candidates)

            best_indices = jnp.argmax(qs, axis=1)

            batch_indices = jnp.arange(batch_size)
            best_actions = actions[batch_indices, best_indices]

            without_residual_mask = best_indices < self.N
            with_residual_mask = (best_indices >= self.N) & (best_indices < total_candidates)
            vf_select_ratio_without_residual = jnp.mean(without_residual_mask.astype(jnp.float32))
            vf_select_ratio_with_residual = jnp.mean(with_residual_mask.astype(jnp.float32))
        else:
            best_actions = actions[:, 0]
            vf_select_ratio_without_residual = 1
            vf_select_ratio_with_residual = 0

        sample_info_extra = {
            "select_ratio_without_residual": vf_select_ratio_without_residual,
            "select_ratio_with_residual": vf_select_ratio_with_residual,
            "mean_d_actions_norm": mean_d_actions_norm,
            "mean_residual_scaled_norm": mean_residual_scaled_norm,
        }

        rng, _ = jax.random.split(rng, 2)
        return jnp.array(best_actions.squeeze()), sample_info_extra, rng
         
    def update_residual_actor(self, batch: DatasetDict) -> Tuple[AgentLearner, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        dropout_key, rng = jax.random.split(rng)

        def residual_actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:

            observations = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, batch["observations"], stop_gradient=True)
            
            # Apply sharding constraint to encoded observations
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)
            dist = self.residual_actor.apply_fn({"params": actor_params}, observations, actions=batch["actions"], training=True, p=batch['states'], rngs={"dropout": dropout_key},)
            actions = dist.sample(seed=key)

            log_probs = dist.log_prob(actions)
            residual_scaled = self._apply_residual_xyzg_mask(actions * self.edit_scale)
            # Subtract log of action scale for each action dimension
            log_probs -= actions.shape[-1] * jnp.log(self.edit_scale)

            actions = residual_scaled + batch["actions"]

            logits = self.critic.apply_fn(
                {"params": self.critic.params, "batch_stats": self.critic.batch_stats},
                observations,
                actions,
                False,
                p=batch['critic_states'],
            )
            # train=False here: this call SCORES the residual actor's
            # proposed action (its output feeds the actor's own gradient) —
            # it doesn't train the critic itself, so it should use the
            # critic's stable running BatchNorm statistics, exactly like
            # every other place the critic is queried (rather than trained)
            # elsewhere in this file (compute_q, sample_actions,
            # sample_batch_actions). No more dropout rng either — the new
            # categorical critic architecture has no dropout layer (XQC's
            # own recipe doesn't include one).
            q = q_from_logits(logits, self.atoms)
            # KL regularization (XQCfD-style: replaces/augments the generic
            # entropy bonus with a penalty for the edit distribution
            # deviating from a fixed reference — "prefer staying close to
            # zero residual unless Q strongly justifies deviating", a
            # learned/adaptive analogue of the hard edit_scale cap).
            # Computed in the PRE-TANH (Gaussian) space, not the squashed
            # action space — KL between two tanh-squashed distributions has
            # no clean closed form (the same reason TFP can't compute
            # entropy() for a TanhTransformedDistribution either, which is
            # what forced the try/except fallback in ppo.py/grpo.py). Two
            # Gaussians DO have a closed-form KL, so we use dist.distribution
            # (the underlying pre-squash MultivariateNormalDiag) directly.
            # kl_coef=0.0 (the default) makes this an exact no-op — same
            # behavior as before this was added.
            base_dist = dist.distribution
            residual_mean = base_dist.mean()
            residual_std = base_dist.stddev()
            kl_per_dim = (
                jnp.log(self.kl_ref_std / residual_std)
                + (residual_std ** 2 + residual_mean ** 2) / (2 * self.kl_ref_std ** 2)
                - 0.5
            )
            kl_penalty = kl_per_dim.sum(axis=-1)
            # Use a manually fixed temperature when configured (bypasses the learned
            # temperature entirely) — a quick diagnostic Jesse suggested to see if the
            # runaway/unconverged alpha behavior is itself contributing to instability,
            # independent of whatever is or isn't miscalibrated in the learned version.
            temperature = (
                self.fixed_temperature if self.fixed_temperature is not None
                else self.temp.apply_fn({"params": self.temp.params})
            )
            residual_actor_loss = (
                self.entropy_scale * log_probs * temperature - q + self.kl_coef * kl_penalty
            ).mean()
            return residual_actor_loss, {
                "residual_q": q.mean(),
                "residual_actor_loss": residual_actor_loss,
                "entropy": -log_probs.mean(),
                "temperature": jnp.asarray(temperature),
                "kl_penalty": kl_penalty.mean(),
                # Pre-tanh Gaussian stats (already computed above for the KL
                # formula itself) — logged so kl_ref_std can be calibrated
                # against the actual equilibrium std under entropy alone,
                # rather than guessed. mean_residual_scaled_norm (elsewhere)
                # is POST-tanh and POST-edit_scale, so it saturates and
                # doesn't directly show this.
                "residual_mean_norm": jnp.linalg.norm(residual_mean, axis=-1).mean(),
                "residual_std_mean": residual_std.mean(),
            }

        grads, actor_info = jax.grad(residual_actor_loss_fn, has_aux=True)(self.residual_actor.params)
        residual_actor = self.residual_actor.apply_gradients(grads=grads)

        return self.replace(residual_actor=residual_actor, rng=rng), actor_info
    
    def update_actor(self, batch: DatasetDict) -> Tuple[AgentLearner, Dict[str, float]]:
        actor_batch = self.actor.prepare_batch_for_actor(batch)
        
        # Ensure actor_batch has correct sharding
        def ensure_sharding(x):
            if isinstance(x, jnp.ndarray):
                return jax.device_put(x, self.data_sharding)
            return x
        actor_batch = jax.tree_util.tree_map(ensure_sharding, actor_batch)

        rng = self.rng
        key, rng = jax.random.split(rng, 2)
        
        with _sharding.set_mesh(self.actor.mesh):
            new_train_state, info = self.actor.train_step(key, self.actor_train_state, actor_batch)

        new_train_state_params = self.actor.get_params(new_train_state)
        target_score_params = optax.incremental_update(
            new_train_state_params, self.target_actor_params, self.actor_tau
        )

        new_agent = self.replace(actor_train_state=new_train_state, target_actor_params=target_score_params, rng=rng)
        
        return new_agent, info


    def update_temperature(self, entropy: float) -> Tuple[AgentLearner, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info


    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        next_actions, sample_info, rng = self.sample_batch_actions(batch)
        next_actions = jax.device_put(next_actions, self.data_sharding)

        key, rng = jax.random.split(rng)

        next_observations = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, 
                                        batch["next_observations"], stop_gradient=True)
        next_observations = jax.device_put(next_observations, self.data_sharding)  

        # No ensemble subsampling anymore (single critic + single target
        # critic — XQCfD's own reported setup uses none). training=False:
        # BatchNorm uses its running mean/var here, not this batch's.
        next_logits = self.target_critic.apply_fn(
            {"params": self.target_critic.params, "batch_stats": self.target_critic.batch_stats},
            next_observations,
            next_actions,
            False,
            p=batch['next_critic_states'],
        )
        next_probs = jax.nn.softmax(next_logits, axis=-1)

        # Same defensive NaN handling as the old scalar critic (there
        # replacing a NaN Q with 0.0) — replace a NaN row's distribution with
        # a uniform one (maximum entropy, Q = midpoint of the support) rather
        # than letting a single bad row poison the whole batch's loss.
        next_probs_nan_mask = jnp.isnan(next_probs).any(axis=-1)
        next_q_nan_ratio = jnp.mean(next_probs_nan_mask)
        uniform_probs = jnp.ones_like(next_probs) / self.num_atoms
        next_probs = jnp.where(next_probs_nan_mask[:, None], uniform_probs, next_probs)

        # Normalize rewards by a running RMS estimate BEFORE the Bellman
        # projection, keeping the fixed [v_min, v_max] support meaningful
        # regardless of this task's absolute reward scale — see create()'s
        # docstring for reward_scale_decay. Uses the scale from BEFORE this
        # batch updates it (below), so there's no within-batch leakage.
        reward_scale = jnp.sqrt(self.reward_ms + 1e-6)
        normalized_rewards = batch["rewards"] / reward_scale

        discount_k = self.discount ** self.replan_steps
        target_probs = categorical_bellman_projection(
            next_probs, normalized_rewards, batch["masks"], discount_k,
            self.atoms, self.v_min, self.v_max,
        )
        target_q = jnp.sum(target_probs * self.atoms, axis=-1)  # E[atoms] in NORMALIZED units, for logging — not part of the loss (the loss uses the full distribution, not this scalar summary)
        target_q_denorm = target_q * reward_scale  # same quantity, rescaled back to this task's raw reward units, purely for human-readable logging

        new_reward_ms = self.reward_scale_decay * self.reward_ms + (1 - self.reward_scale_decay) * jnp.mean(batch["rewards"] ** 2)

        key, rng = jax.random.split(rng)

        params_dict = {"critic": self.critic.params}
        if not self.freeze_critic_encoder:
            params_dict["batch_encoder"] = self.batch_encoder.params

        def critic_loss_fn(params_dict) -> Tuple[jnp.ndarray, Dict[str, float]]:
            if self.freeze_critic_encoder:
                observations = batch_encode(
                    self.batch_encoder.apply_fn,
                    self.batch_encoder.params,
                    batch["observations"],
                    stop_gradient=True,
                )
            else:
                observations = batch_encode(
                    self.batch_encoder.apply_fn, params_dict["batch_encoder"], batch["observations"]
                )
            # Apply sharding constraint to encoded observations (works inside grad)
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)
            # Per-sample L2 norm of the encoder's output feature vector — the
            # thing that feeds directly into the critic's Q computation.
            encoder_feature_norms = jnp.linalg.norm(observations, axis=-1)
            # training=True: BatchNorm uses (and updates) this batch's own
            # statistics. mutable=['batch_stats'] returns the updated running
            # mean/var alongside the logits — captured here as `model_state`
            # and threaded back onto the critic TrainState AFTER
            # jax.grad returns (batch_stats are a forward-pass running
            # average, not something to differentiate w.r.t. params).
            logits, model_state = self.critic.apply_fn(
                {"params": params_dict['critic'], "batch_stats": self.critic.batch_stats},
                observations,
                batch["actions"],
                True,
                p=batch['critic_states'],
                mutable=['batch_stats'],
            )
            per_sample_loss = categorical_cross_entropy_loss(logits, target_probs)
            critic_loss = (per_sample_loss * batch["valids"]).mean()
            q = q_from_logits(logits, self.atoms)
            q_denorm = q * reward_scale  # rescaled back to raw reward units, for human-readable logging only
            return critic_loss, {
                "critic_loss": critic_loss,
                "q": q.mean(),
                "q_min": q.min(),
                "q_max": q.max(),
                "target_q_min": target_q.min(),
                "target_q_max": target_q.max(),
                "target_q_mean": target_q.mean(),
                # Same quantities rescaled back to this task's raw reward
                # units — NOT bounded (reward_scale itself changes over
                # training), purely for human interpretability. The
                # normalized q/target_q above are the ones that are
                # structurally guaranteed to stay within [v_min, v_max].
                "q_denorm": q_denorm.mean(),
                "q_min_denorm": q_denorm.min(),
                "q_max_denorm": q_denorm.max(),
                "target_q_max_denorm": target_q_denorm.max(),
                "target_q_min_denorm": target_q_denorm.min(),
                "reward_scale": reward_scale,
                "encoder_feature_norm_mean": encoder_feature_norms.mean(),
                "encoder_feature_norm_max": encoder_feature_norms.max(),
                "_new_critic_batch_stats": model_state["batch_stats"],
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(params_dict)
        new_critic_batch_stats = info.pop("_new_critic_batch_stats")

        critic = self.critic.apply_gradients(grads=grads["critic"])
        # XQC's post-optimizer-step weight normalization: project every
        # Dense kernel's columns back to unit L2 norm, AFTER the gradient
        # step (not part of the gradient computation itself).
        critic = critic.replace(
            params=project_weights_to_unit_norm(critic.params),
            batch_stats=new_critic_batch_stats,
        )

        if self.freeze_critic_encoder:
            batch_encoder = self.batch_encoder
        else:
            batch_encoder = self.batch_encoder.apply_gradients(grads=grads["batch_encoder"]) 

        critic_grad_norm = optax.global_norm(grads["critic"])
        info["critic_grad_norm"] = critic_grad_norm
        # The line above logs the RAW gradient norm — computed from `grads`
        # BEFORE `apply_gradients` (above) runs it through the clipping
        # transform internally. It will show values above the clip threshold
        # even when clipping is working correctly; it only tells you what the
        # gradient looked like pre-clip, not what actually got applied.
        # This second metric recomputes the clip explicitly (pure, no side
        # effects on the real update) so the two can be compared directly —
        # post-clip should hard-ceiling at critic_grad_clip_norm whenever
        # pre-clip exceeds it.
        if self.critic_grad_clip_norm is not None:
            _clip_tx = optax.clip_by_global_norm(self.critic_grad_clip_norm)
            _clipped_critic_grads, _ = _clip_tx.update(grads["critic"], _clip_tx.init(grads["critic"]))
            info["critic_grad_norm_post_clip"] = optax.global_norm(_clipped_critic_grads)
        else:
            info["critic_grad_norm_post_clip"] = critic_grad_norm
        info["critic_param_norm"] = optax.global_norm(critic.params)
        info["next_q_nan_ratio"] = next_q_nan_ratio
        
        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic_batch_stats = optax.incremental_update(
            critic.batch_stats, self.target_critic.batch_stats, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params, batch_stats=target_critic_batch_stats)
        info["target_critic_param_norm"] = optax.global_norm(target_critic_params)

        info.update(sample_info)

        info["reward_ms"] = new_reward_ms

        return self.replace(critic=critic, target_critic=target_critic, batch_encoder=batch_encoder, reward_ms=new_reward_ms, rng=rng), info


    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        # Avoid letting actor_batch flip between None and a real dict across
        # the jax.jit boundary — that's a different pytree structure each
        # time, and forces JAX to trace/compile (and keep resident) a SEPARATE
        # XLA program the first time a successful episode appears in the
        # buffer, potentially well into training (e.g. the OOM we hit around
        # step 2473 on a run that started with 0% base success). Pass a
        # static bool instead, and always give _update_jit a
        # consistently-shaped placeholder (reusing `batch`'s own structure,
        # which goes through the same prepare_critic_batch() call) when no
        # real success-only actor_batch is available yet.
        use_success_batch = actor_batch is not None
        if actor_batch is None:
            actor_batch = batch
        # Drop stale inference copies before JIT; rebuild after so rollouts use new weights.
        new_agent, info = self.replace(_infer_cache=None)._update_jit(
            agent.replace(_infer_cache=None), batch, utd_ratio, actor_batch, use_success_batch
        )
        return new_agent.cache_infer_params(), info


    @partial(jax.jit, static_argnames=("utd_ratio", "use_success_batch"))
    def _update_jit(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None, use_success_batch: bool = False):
        batch = batch.copy()
        rng, key1 = jax.random.split(agent.rng)
        rng, key2 = jax.random.split(rng)
        batch["image"] = self.data_augmentation_fn(key1, batch["image"])
        batch["next_image"] = self.data_augmentation_fn(key2, batch["next_image"])
        batch = prepare_critic_batch(batch, self.actor.model_config.action_dim, self.action_dim, self.state_dim, self.action_horizon, self.replan_steps)
        new_agent = agent.replace(rng=rng)

        total_bs = batch["actions"].shape[0]
        assert total_bs % utd_ratio == 0, (
            f"Batch size ({total_bs}) must be a multiple of utd_ratio ({utd_ratio})"
        )
        minibatch_size = total_bs // utd_ratio

        def reshape_minibatch(x):
            return x.reshape((utd_ratio, minibatch_size) + x.shape[1:])

        minibatches = jax.tree_util.tree_map(reshape_minibatch, batch)

        def create_minibatch_sharding(x):
            # Create sharding spec: (None, DATA_AXIS, ...)
            # None for utd_ratio dimension, DATA_AXIS for minibatch dimension
            ndim = len(x.shape)
            spec_tuple = (None,) + (_sharding.DATA_AXIS,) + (None,) * (ndim - 2)
            new_sharding = jax.sharding.NamedSharding(
                self.data_sharding.mesh,
                jax.sharding.PartitionSpec(*spec_tuple)
            )
            return jax.device_put(x, new_sharding)

        minibatches = jax.tree_util.tree_map(create_minibatch_sharding, minibatches)

        def critic_update_step(carry, mb):
            (agent,) = carry
            agent, info = agent.update_critic(mb)
            return (agent,), info

        (new_agent,), critic_infos = jax.lax.scan(critic_update_step, (new_agent,), minibatches)

        # Use last minibatch for actor updates
        last_minibatch = jax.tree_util.tree_map(lambda x: x[-1] if x is not None and hasattr(x, "shape") else x, minibatches)

        # When actor_success_only, use the dedicated success-episode batch for
        # the Pi05 actor update; otherwise (or if no successful episode exists
        # yet in the buffer, e.g. early in training / near-0% base success)
        # fall back to the last critic minibatch.
        if self.actor_success_only and use_success_batch:
            actor_batch = actor_batch.copy()
            rng, key = jax.random.split(new_agent.rng)
            actor_batch["image"] = self.data_augmentation_fn(key, actor_batch["image"])
            new_agent = new_agent.replace(rng=rng)
            actor_batch = prepare_critic_batch(actor_batch, self.actor.model_config.action_dim, self.action_dim, self.state_dim, self.action_horizon, self.replan_steps)
            new_agent, actor_info = new_agent.update_actor(actor_batch)
        else:
            new_agent, actor_info = new_agent.update_actor(last_minibatch)

        actor_info = dict(actor_info)

        if self.n_edit_samples > 0:
            new_agent, r_actor_info = new_agent.update_residual_actor(last_minibatch)
            actor_info = {**actor_info, **r_actor_info}
            if self.fixed_temperature is None:
                new_agent, temp_info = new_agent.update_temperature(r_actor_info["entropy"])
                actor_info = {**actor_info, **temp_info}
            else:
                # Report the fixed value too, so it still shows up alongside the
                # learned-temperature runs in wandb/TensorBoard for comparison.
                actor_info["temperature"] = jnp.asarray(self.fixed_temperature)

        critic_info = jax.tree_util.tree_map(lambda x: x[-1], critic_infos)
        return new_agent, {**actor_info, **critic_info}
