"""GRPOLearner: Group Relative Policy Optimization, from pixels.

Like PPOLearner, this is on-policy. Unlike PPO, GRPO has NO value network /
critic: instead of a learned baseline, the advantage for each rollout is
computed relative to a GROUP of other rollouts of the same task instance
(e.g. several attempts from the same initial episode seed), normalized by
the group's own mean and std:

    advantage_i = (return_i - mean(returns_in_group)) / (std(returns_in_group) + eps)

This removes the need for a value function entirely, at the cost of needing
G > 1 rollouts per group to get a meaningful baseline, and of typically being
less sample-efficient per rollout than a learned critic. Because there's no
critic to keep the policy in check, GRPO implementations (this one included)
usually add an explicit KL-divergence penalty against a fixed reference
policy snapshot to prevent the policy from drifting arbitrarily far in a
single training run — this is the standard mitigation in the GRPO literature
this algorithm originates from (DeepSeekMath / DeepSeek-R1 style RL).

IMPORTANT — what `batch` must look like, on top of the on-policy requirement
already described in ppo.py's module docstring:
  - `batch` must be organized so that `group_size` consecutive entries along
    the batch axis belong to the same group (same initial episode seed /
    task instance, `group_size` independent rollouts each) — i.e. shape
    (num_groups * group_size, ...) with groups contiguous. This is a rollout-
    collection requirement outside this file's scope (see module docstring
    in the pipeline integration when that happens).
  - a per-transition `episode_return` field (the FULL, undiscounted return of
    the episode that transition belongs to) is required, since the
    group-relative advantage is computed per-episode, not per-transition —
    every transition within one episode shares the same episode_return and
    therefore the same advantage.

`utd_ratio` is reinterpreted as the number of GRPO epochs (K) over the given
rollout batch, matching PPOLearner's convention and the existing pipeline's
calling pattern.

No residual/edit mechanism, no frozen VLA action-head reliance. The VLA `vla`
object is used only for input preprocessing and output denormalization.
"""

from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import dataclasses

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import struct
from flax.training.train_state import TrainState

import openpi.shared.array_typing as at
import openpi.training.sharding as _sharding

from expo_ft.agents.alg.agent import AgentLearner, initialize_checkpoint_dir
from expo_ft.agents.alg.checkpoint_utils import make_checkpoint_fns
from expo_ft.agents.alg.batch_utils import prepare_critic_batch
from expo_ft.data.dataset import DatasetDict
from expo_ft.distributions import TanhNormal
from expo_ft.networks import MLP, BatchEncoder
from expo_ft.networks.pixel_multiplexer import PixelTanhNormalMultiplexer
from expo_ft.networks.encoders import ResNetV2Encoder
from expo_ft.utils.augmentation import make_data_augmentation_fn


def _split_params(agent: Any) -> tuple[Any, dict[str, at.Params]]:
    batch_encoder_params = agent.batch_encoder.params
    actor_params = agent.actor.params
    ref_actor_params = agent.ref_actor_params

    agent = dataclasses.replace(
        agent,
        batch_encoder=dataclasses.replace(agent.batch_encoder, params={}),
        actor=dataclasses.replace(agent.actor, params={}),
        ref_actor_params={},
    )
    params = {
        "batch_encoder_params": batch_encoder_params,
        "actor_params": actor_params,
        "ref_actor_params": ref_actor_params,
    }
    return agent, params


def _merge_params(agent: Any, params: dict[str, at.Params]) -> Any:
    batch_encoder = dataclasses.replace(agent.batch_encoder, params=params["batch_encoder_params"])
    actor = dataclasses.replace(agent.actor, params=params["actor_params"])
    return dataclasses.replace(agent, batch_encoder=batch_encoder, actor=actor, ref_actor_params=params["ref_actor_params"])


_restore_checkpoint, _save_checkpoint = make_checkpoint_fns(_split_params, _merge_params)


def restore_checkpoint(checkpoint_manager, agent, step: int | None = None):
    return _restore_checkpoint(checkpoint_manager, agent, step)


def save_checkpoint(checkpoint_manager: ocp.CheckpointManager, agent: Any, step: int):
    _save_checkpoint(checkpoint_manager, agent, step)


def load_agent(seed, example_observation, example_action, example_state,
                actor, actor_train_state, target_actor_params, agent_kwargs, metadata,
                mesh, data_sharding, replicated_sharding, resume, replan_steps,
                default_prompt, **kwargs):
    """Create a GRPOLearner. `actor` is the pre-built VLA, used only for input
    preprocessing / output denormalization — see module docstring."""
    agent_kwargs.update(
        vla=actor,
        mesh=mesh,
        resume=resume,
        replan_steps=replan_steps,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        default_prompt=default_prompt,
        **metadata,
    )
    return GRPOLearner.create(seed, example_observation, example_action, example_state, **agent_kwargs)


@partial(jax.jit, static_argnames=("encoder_fn", "stop_gradient"))
def batch_encode(encoder_fn, encoder_params, observations, stop_gradient=False):
    return encoder_fn({"params": encoder_params}, observations, stop_gradient=stop_gradient)


def compute_group_relative_advantage(episode_returns: jnp.ndarray, group_size: int, eps: float = 1e-6):
    """episode_returns: (num_groups * group_size,), groups contiguous.
    Returns advantages of the same shape, one value per transition (constant within an episode)."""
    grouped = episode_returns.reshape(-1, group_size)
    group_mean = grouped.mean(axis=1, keepdims=True)
    group_std = grouped.std(axis=1, keepdims=True)
    advantages = (grouped - group_mean) / (group_std + eps)
    return advantages.reshape(-1)


class GRPOLearner(AgentLearner, struct.PyTreeNode):
    """On-policy GRPO: stochastic policy only (no critic), group-relative advantage,
    clipped surrogate objective, KL penalty against a fixed reference policy snapshot."""

    rng: jax.random.PRNGKey
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    vla: Any = struct.field(pytree_node=False)
    batch_encoder: TrainState
    actor: TrainState
    ref_actor_params: at.Params  # frozen snapshot for the KL penalty; refreshed periodically by the caller
    group_size: int = struct.field(pytree_node=False)
    clip_eps: float
    kl_coef: float
    entropy_coef: float
    max_grad_norm: Optional[float] = struct.field(pytree_node=False)
    num_minibatches: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    full_action_dim: int = struct.field(pytree_node=False)
    replan_steps: int = struct.field(pytree_node=False)
    action_horizon: int = struct.field(pytree_node=False)
    resize_size: Optional[int] = struct.field(pytree_node=False)
    default_prompt: Optional[str] = struct.field(pytree_node=False)
    data_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    _infer_cache: Optional[dict] = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        states,
        vla: Any = None,
        action_horizon: int = 1,
        mesh: Optional[Any] = None,
        # GRPO hyperparameters
        actor_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256, 256),
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_coef: float = 0.04,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        num_minibatches: int = 4,
        use_pnorm: bool = False,
        actor_drop: Optional[float] = None,
        include_state: bool = True,
        latent_dim_image: int = 50,
        latent_dim_state: int = 50,
        encoder_stage_sizes: Tuple[int, int, int, int] = (2, 2, 2, 2),
        encoder_num_filters: int = 64,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
        resume: bool = False,
        replan_steps: int = 1,
        data_sharding: Optional[jax.sharding.NamedSharding] = None,
        replicated_sharding: Optional[jax.sharding.NamedSharding] = None,
        default_prompt: Optional[str] = None,
        resize_size: Optional[int] = None,
        use_full_augmentation: bool = True,
        **kwargs,
    ):
        action_dim = action_space.shape[-1]
        state_dim = states.shape[-1]
        full_action_dim = replan_steps * action_dim
        actions = jnp.zeros((full_action_dim,))
        print("[GRPOLearner] observation shape:", observation_space.shape)
        print("[GRPOLearner] action shape:", actions.shape, "action horizon:", action_horizon, "action_dim:", action_dim)
        print("[GRPOLearner] states shape:", states.shape, "group_size:", group_size)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, encoder_key = jax.random.split(rng, 3)

        encoder_cls = partial(ResNetV2Encoder, stage_sizes=encoder_stage_sizes, num_filters=encoder_num_filters)
        batch_encoder_def = BatchEncoder(
            encoder_cls=encoder_cls, latent_dim=latent_dim_image, pixel_keys=pixel_keys, depth_keys=depth_keys
        )
        batch_encoder_params = batch_encoder_def.init(encoder_key, observation_space)["params"]
        batch_encoder = TrainState.create(
            apply_fn=batch_encoder_def.apply, params=batch_encoder_params, tx=optax.adam(learning_rate=actor_lr)
        )
        batch_encoder_shape = jax.eval_shape(lambda: batch_encoder)
        batch_encoder_sharding = _sharding.fsdp_sharding(batch_encoder_shape, mesh, log=True)
        batch_encoder = jax.jit(
            lambda x: x, in_shardings=replicated_sharding, out_shardings=batch_encoder_sharding
        )(batch_encoder)

        critic_observations = jnp.ones((1, latent_dim_image))
        critic_states = jnp.expand_dims(states, axis=0)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm)
        actor_dist_cls = TanhNormal(actor_base_cls, full_action_dim)
        actor_def = PixelTanhNormalMultiplexer(
            network_cls=actor_dist_cls, latent_dim=latent_dim_image, include_state=include_state,
            state_latent_dim=latent_dim_state,
        )
        actor_params = actor_def.init(actor_key, critic_observations, p=critic_states)["params"]
        actor_tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm) if max_grad_norm is not None else optax.identity(),
            optax.adam(learning_rate=actor_lr),
        )
        actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=actor_tx)
        actor_shape = jax.eval_shape(lambda: actor)
        actor_sharding = _sharding.fsdp_sharding(actor_shape, mesh, log=True)
        actor = jax.jit(lambda x: x, in_shardings=replicated_sharding, out_shardings=actor_sharding)(actor)

        # Reference policy for the KL penalty starts as a copy of the initial actor.
        # The caller is responsible for periodically refreshing it (e.g. every N updates)
        # by replacing `ref_actor_params` with a snapshot of `actor.params` — this file
        # does not do that automatically, since the right refresh cadence is a training-
        # loop decision, not an algorithm-internals one.
        ref_actor_params = actor_params

        agent = cls(
            rng=rng,
            data_augmentation_fn=make_data_augmentation_fn(use_full_augmentation),
            vla=vla,
            batch_encoder=batch_encoder,
            actor=actor,
            ref_actor_params=ref_actor_params,
            group_size=group_size,
            clip_eps=clip_eps,
            kl_coef=kl_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            num_minibatches=num_minibatches,
            action_dim=action_dim,
            state_dim=state_dim,
            full_action_dim=full_action_dim,
            replan_steps=replan_steps,
            action_horizon=action_horizon,
            resize_size=resize_size,
            default_prompt=default_prompt,
            data_sharding=data_sharding,
        )
        if not resume:
            agent = agent.cache_infer_params()
        return agent

    def cache_infer_params(self):
        s = self.vla.infer_sharding
        return self.replace(_infer_cache={
            "batch_encoder_params": jax.device_put(self.batch_encoder.params, s),
            "actor_params": jax.device_put(self.actor.params, s),
        })

    def sample_actions(self, observations, only_base_actions=False):
        """`only_base_actions` accepted for interface parity, no effect (no separate base action)."""
        infer_sharding = self.vla.infer_sharding
        rng = jax.device_put(self.rng, infer_sharding)
        c = self._infer_cache or {}
        _batch_encoder_params = c.get("batch_encoder_params") or jax.device_put(self.batch_encoder.params, infer_sharding)
        _actor_params = c.get("actor_params") or jax.device_put(self.actor.params, infer_sharding)

        transformed_inputs = self.vla.process_raw_inputs(observations, self.action_dim, self.resize_size)
        critic_obs = jnp.concatenate(
            [transformed_inputs["image"]["base_0_rgb"], transformed_inputs["image"]["left_wrist_0_rgb"]], axis=-1
        )
        critic_states = transformed_inputs["state"][..., : self.state_dim]

        key, rng = jax.random.split(rng)
        encoded_obs = batch_encode(self.batch_encoder.apply_fn, _batch_encoder_params, critic_obs, stop_gradient=True)
        dist = self.actor.apply_fn({"params": _actor_params}, encoded_obs, p=critic_states)
        action = dist.sample(seed=key).reshape(self.full_action_dim)

        action = action.reshape(1, self.replan_steps, self.action_dim)
        # Pad only the horizon dimension here (replan_steps -> action_horizon) —
        # process_transformed_outputs() already pads the action dimension
        # itself internally (env action_dim -> the VLA model's padded
        # action_dim). Padding the action dimension here too (to
        # self.vla.model_config.action_dim) double-pads it, producing a
        # flattened size process_transformed_outputs can't reshape back
        # (e.g. "cannot reshape array of shape (1, 512) into shape (1, 16, 8)").
        padded = jnp.zeros((1, self.action_horizon, self.action_dim)).at[:, : self.replan_steps, :].set(action)
        raw_action = self.vla.process_transformed_outputs(padded)[0]
        n = min(self.replan_steps, self.action_horizon)
        action = raw_action[:n].reshape(n, self.action_dim)
        sample_info = {"sample_time": 0.0}
        return jnp.array(action), self.replace(rng=rng), sample_info

    def update_actor(self, batch: DatasetDict) -> Tuple["GRPOLearner", Dict[str, float]]:
        """Single-minibatch GRPO actor update. Expects `batch` to already carry
        `old_log_probs`, `advantages` (see `update()`, which computes these before
        calling this per-minibatch)."""
        def loss_fn(params):
            observations = batch_encode(self.batch_encoder.apply_fn, params["batch_encoder"], batch["observations"])
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)

            dist = self.actor.apply_fn({"params": params["actor"]}, observations, p=batch["states"])
            log_probs = dist.log_prob(batch["actions"])
            ratio = jnp.exp(log_probs - batch["old_log_probs"])

            adv = batch["advantages"]
            surr1 = ratio * adv
            surr2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_loss = -jnp.minimum(surr1, surr2).mean()

            # KL penalty against the fixed reference policy (per-token/per-sample
            # log-ratio estimator, the standard low-variance GRPO KL estimator:
            # KL(pi || pi_ref) ~= exp(log_pi_ref - log_pi) - (log_pi_ref - log_pi) - 1).
            ref_dist = self.actor.apply_fn({"params": self.ref_actor_params}, observations, p=batch["states"])
            ref_log_probs = ref_dist.log_prob(batch["actions"])
            log_ratio_ref = ref_log_probs - log_probs
            kl_penalty = (jnp.exp(log_ratio_ref) - log_ratio_ref - 1.0).mean()

            # hasattr(dist, "entropy") only checks the METHOD exists, not that
            # calling it succeeds — TanhNormal is a TFP TransformedDistribution,
            # which always exposes .entropy() (inherited from the base
            # Distribution class) but raises NotImplementedError when called,
            # since a Tanh-squashed Gaussian has no closed-form entropy. Use
            # try/except to actually test computability; this resolves once
            # during tracing (dist's type is static, not a traced value), so
            # it's exactly as cheap as the hasattr check it replaces — just
            # correct. Falls back to the standard sample-based entropy proxy
            # (-log_probs.mean()) used throughout this codebase for exactly
            # this situation.
            try:
                entropy = dist.entropy().mean()
            except NotImplementedError:
                entropy = -log_probs.mean()
            loss = policy_loss + self.kl_coef * kl_penalty - self.entropy_coef * entropy

            approx_kl = (batch["old_log_probs"] - log_probs).mean()
            clip_frac = (jnp.abs(ratio - 1.0) > self.clip_eps).mean().astype(jnp.float32)

            return loss, {
                "grpo_loss": loss, "policy_loss": policy_loss, "kl_penalty": kl_penalty,
                "entropy": entropy, "approx_kl": approx_kl, "clip_frac": clip_frac,
                "ratio_mean": ratio.mean(),
            }

        params = {"actor": self.actor.params, "batch_encoder": self.batch_encoder.params}
        grads, info = jax.grad(loss_fn, has_aux=True)(params)

        actor = self.actor.apply_gradients(grads=grads["actor"])
        batch_encoder = self.batch_encoder.apply_gradients(grads=grads["batch_encoder"])
        info["actor_param_norm"] = optax.global_norm(actor.params)

        return self.replace(actor=actor, batch_encoder=batch_encoder), info

    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        """`utd_ratio` is the number of GRPO epochs (K) over this on-policy rollout batch.
        `actor_batch` is unused — accepted only for interface parity with the pipeline.
        `batch` must contain `episode_returns` (per-transition, constant within an episode)
        and be organized with `group_size` contiguous rollouts per group — see module
        docstring."""
        new_agent, info = self.replace(_infer_cache=None)._update_jit(
            agent.replace(_infer_cache=None), batch, utd_ratio,
        )
        return new_agent.cache_infer_params(), info

    @partial(jax.jit, static_argnames=("num_epochs",))
    def _update_jit(self, agent, batch: DatasetDict, num_epochs: int):
        batch = batch.copy()
        rng, key1 = jax.random.split(agent.rng)
        batch["image"] = self.data_augmentation_fn(key1, batch["image"])
        batch = prepare_critic_batch(batch, self.vla.model_config.action_dim, self.action_dim, self.state_dim, self.action_horizon, self.replan_steps)

        encoded_obs = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, batch["observations"], stop_gradient=True)
        dist = self.actor.apply_fn({"params": self.actor.params}, encoded_obs, p=batch["states"])
        old_log_probs = dist.log_prob(batch["actions"])

        advantages = compute_group_relative_advantage(batch["episode_returns"], self.group_size)

        batch["old_log_probs"] = old_log_probs
        batch["advantages"] = advantages

        rng, key2 = jax.random.split(rng)
        new_agent = agent.replace(rng=rng)

        total_bs = batch["actions"].shape[0]
        assert total_bs % self.num_minibatches == 0, (
            f"Rollout batch size ({total_bs}) must be a multiple of num_minibatches ({self.num_minibatches})"
        )
        minibatch_size = total_bs // self.num_minibatches

        def epoch_step(carry, key):
            (agent,) = carry
            perm = jax.random.permutation(key, total_bs)
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], batch)

            def reshape_minibatch(x):
                return x.reshape((self.num_minibatches, minibatch_size) + x.shape[1:])

            minibatches = jax.tree_util.tree_map(reshape_minibatch, shuffled)

            def minibatch_step(carry, mb):
                (agent,) = carry
                agent, info = agent.update_actor(mb)
                return (agent,), info

            (agent,), infos = jax.lax.scan(minibatch_step, (agent,), minibatches)
            last_info = jax.tree_util.tree_map(lambda x: x[-1], infos)
            return (agent,), last_info

        epoch_keys = jax.random.split(key2, num_epochs)
        (new_agent,), epoch_infos = jax.lax.scan(epoch_step, (new_agent,), epoch_keys)
        info = jax.tree_util.tree_map(lambda x: x[-1], epoch_infos)

        return new_agent, info
