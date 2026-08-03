"""GRPOResidualLearner: on-policy GRPO, but trained on a RESIDUAL policy
conditioned on the frozen base VLA's own action -- the GRPO counterpart of
ppo_residual.py's PPOResidualLearner. See that module's docstring for the
full rationale (identical here): pi0.5's own log-probability is never
needed, only the residual's own (tractable, TanhNormal) one.

Same relationship to grpo.py as PPOResidualLearner has to ppo.py: no value
network (GRPO has none), group-relative advantage computed from
`episode_returns` + `group_size`, KL penalty against a frozen reference
snapshot of the RESIDUAL actor (not the VLA, which is never updated and
needs no such penalty).

IMPORTANT -- what `batch` must additionally contain, on top of grpo.py's own
on-policy + group requirements (see that module's docstring):
  - `base_actions`: same meaning, same normalized space, same rollout-time-
    not-resampled requirement as ppo_residual.py's docstring describes.
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
from expo_ft.agents.alg.grpo import compute_group_relative_advantage, batch_encode
from expo_ft.data.dataset import DatasetDict
from expo_ft.distributions import TanhNormal
from expo_ft.networks import MLP, BatchEncoder
from expo_ft.networks.pixel_multiplexer import PixelEditMultiplexer
from expo_ft.networks.encoders import ResNetV2Encoder
from expo_ft.utils.augmentation import make_data_augmentation_fn


def _split_params(agent: Any) -> tuple[Any, dict[str, at.Params]]:
    batch_encoder_params = agent.batch_encoder.params
    residual_actor_params = agent.residual_actor.params
    ref_residual_actor_params = agent.ref_residual_actor_params

    agent = dataclasses.replace(
        agent,
        batch_encoder=dataclasses.replace(agent.batch_encoder, params={}),
        residual_actor=dataclasses.replace(agent.residual_actor, params={}),
        ref_residual_actor_params={},
    )
    params = {
        "batch_encoder_params": batch_encoder_params,
        "residual_actor_params": residual_actor_params,
        "ref_residual_actor_params": ref_residual_actor_params,
    }
    return agent, params


def _merge_params(agent: Any, params: dict[str, at.Params]) -> Any:
    batch_encoder = dataclasses.replace(agent.batch_encoder, params=params["batch_encoder_params"])
    residual_actor = dataclasses.replace(agent.residual_actor, params=params["residual_actor_params"])
    return dataclasses.replace(
        agent, batch_encoder=batch_encoder, residual_actor=residual_actor,
        ref_residual_actor_params=params["ref_residual_actor_params"],
    )


_restore_checkpoint, _save_checkpoint = make_checkpoint_fns(_split_params, _merge_params)


def restore_checkpoint(checkpoint_manager, agent, step: int | None = None):
    return _restore_checkpoint(checkpoint_manager, agent, step)


def save_checkpoint(checkpoint_manager: ocp.CheckpointManager, agent: Any, step: int):
    _save_checkpoint(checkpoint_manager, agent, step)


def load_agent(seed, example_observation, example_action, example_state,
                actor, actor_train_state, target_actor_params, agent_kwargs, metadata,
                mesh, data_sharding, replicated_sharding, resume, replan_steps,
                default_prompt, **kwargs):
    """Create a GRPOResidualLearner. `actor_train_state` IS used here (to
    sample the frozen VLA's own base action) -- see ppo_residual.py's
    load_agent docstring, identical rationale."""
    agent_kwargs.update(
        vla=actor,
        actor_train_state=actor_train_state,
        mesh=mesh,
        resume=resume,
        replan_steps=replan_steps,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        default_prompt=default_prompt,
        **metadata,
    )
    return GRPOResidualLearner.create(seed, example_observation, example_action, example_state, **agent_kwargs)


class GRPOResidualLearner(AgentLearner, struct.PyTreeNode):
    """On-policy GRPO trained on a residual policy conditioned on the frozen
    base VLA's own sampled action. See module docstring."""

    rng: jax.random.PRNGKey
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    vla: Any = struct.field(pytree_node=False)
    actor_train_state: Any = struct.field(pytree_node=False)  # frozen VLA's own train_state, sampling only -- never updated
    batch_encoder: TrainState
    residual_actor: TrainState
    ref_residual_actor_params: at.Params  # frozen snapshot of the RESIDUAL actor for the KL penalty; refreshed periodically by the caller
    edit_scale: float = struct.field(pytree_node=False)
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
        actor_train_state: Any = None,
        action_horizon: int = 1,
        mesh: Optional[Any] = None,
        # GRPO hyperparameters (identical defaults to GRPOLearner, for comparability)
        actor_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256, 256),
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_coef: float = 0.04,
        entropy_coef: float = 0.01,
        actor_log_std_min: float = -5.0,
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
        # Residual-specific
        edit_scale: float = 0.2,
        **kwargs,
    ):
        action_dim = action_space.shape[-1]
        state_dim = states.shape[-1]
        full_action_dim = replan_steps * action_dim
        actions = jnp.zeros((full_action_dim,))
        print("[GRPOResidualLearner] observation shape:", observation_space.shape)
        print("[GRPOResidualLearner] action shape:", actions.shape, "action horizon:", action_horizon, "action_dim:", action_dim)
        print("[GRPOResidualLearner] states shape:", states.shape, "group_size:", group_size, "edit_scale:", edit_scale)

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
        critic_actions = jnp.expand_dims(actions, axis=0)

        # Residual actor: identical construction to ppo_residual.py's own
        # (and to ExpoFT's), conditioned on the base action via
        # PixelEditMultiplexer. Sampled directly, no critic involved.
        residual_actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm
        )
        residual_actor_cls = TanhNormal(residual_actor_base_cls, full_action_dim, log_std_min=actor_log_std_min)
        residual_actor_def = PixelEditMultiplexer(
            network_cls=residual_actor_cls,
            latent_dim=latent_dim_image,
            include_state=include_state,
        )
        residual_actor_params = residual_actor_def.init(
            actor_key, critic_observations, actions=critic_actions, p=critic_states
        )["params"]
        actor_tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm) if max_grad_norm is not None else optax.identity(),
            optax.adam(learning_rate=actor_lr),
        )
        residual_actor = TrainState.create(apply_fn=residual_actor_def.apply, params=residual_actor_params, tx=actor_tx)
        residual_actor_shape = jax.eval_shape(lambda: residual_actor)
        residual_actor_sharding = _sharding.fsdp_sharding(residual_actor_shape, mesh, log=True)
        residual_actor = jax.jit(
            lambda x: x, in_shardings=replicated_sharding, out_shardings=residual_actor_sharding
        )(residual_actor)

        # Reference RESIDUAL policy for the KL penalty starts as a copy of
        # the initial residual actor -- the VLA itself is frozen and never
        # updated, so it needs no such reference/penalty. The caller is
        # responsible for periodically refreshing this (see GRPOLearner's
        # own create() docstring note -- same convention here).
        ref_residual_actor_params = residual_actor_params

        agent = cls(
            rng=rng,
            data_augmentation_fn=make_data_augmentation_fn(use_full_augmentation),
            vla=vla,
            actor_train_state=actor_train_state,
            batch_encoder=batch_encoder,
            residual_actor=residual_actor,
            ref_residual_actor_params=ref_residual_actor_params,
            edit_scale=edit_scale,
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
            "residual_actor_params": jax.device_put(self.residual_actor.params, s),
        })

    def sample_actions(self, observations, only_base_actions=False):
        """Identical mechanism to PPOResidualLearner.sample_actions -- see
        that docstring. `only_base_actions=True` executes the raw base
        action, skipping the residual entirely."""
        infer_sharding = self.vla.infer_sharding
        rng = jax.device_put(self.rng, infer_sharding)
        c = self._infer_cache or {}
        _batch_encoder_params = c.get("batch_encoder_params") or jax.device_put(self.batch_encoder.params, infer_sharding)
        _residual_actor_params = c.get("residual_actor_params") or jax.device_put(self.residual_actor.params, infer_sharding)

        transformed_inputs = self.vla.process_raw_inputs(observations, self.action_dim, self.resize_size)
        critic_obs = jnp.concatenate(
            [transformed_inputs["image"]["base_0_rgb"], transformed_inputs["image"]["left_wrist_0_rgb"]], axis=-1
        )
        critic_states = transformed_inputs["state"][..., : self.state_dim]

        key, rng = jax.random.split(rng)
        base_action_full, _ = self.vla.sample_training_actions(
            transformed_inputs=transformed_inputs,
            train_state=self.actor_train_state,
            rng=key,
            train=False,
            num_samples=1,
        )
        base_action = base_action_full[:, : self.replan_steps, :].reshape(1, self.full_action_dim)

        key, rng = jax.random.split(rng)
        encoded_obs = batch_encode(self.batch_encoder.apply_fn, _batch_encoder_params, critic_obs, stop_gradient=True)

        if only_base_actions:
            action = base_action.reshape(self.full_action_dim)
        else:
            dist = self.residual_actor.apply_fn(
                {"params": _residual_actor_params}, encoded_obs, actions=base_action, p=critic_states
            )
            residual = dist.sample(seed=key).reshape(1, self.full_action_dim)
            action = (base_action + self.edit_scale * residual).reshape(self.full_action_dim)

        action_reshaped = action.reshape(1, self.replan_steps, self.action_dim)
        padded = jnp.zeros((1, self.action_horizon, self.action_dim)).at[:, : self.replan_steps, :].set(action_reshaped)
        raw_action = self.vla.process_transformed_outputs(padded)[0]
        n = min(self.replan_steps, self.action_horizon)
        final_action = raw_action[:n].reshape(n, self.action_dim)

        sample_info = {"sample_time": 0.0, "base_action": base_action.reshape(self.full_action_dim)}
        return jnp.array(final_action), self.replace(rng=rng), sample_info

    def update_actor(self, batch: DatasetDict) -> Tuple["GRPOResidualLearner", Dict[str, float]]:
        """Single-minibatch GRPO update, on the recovered residual only --
        see ppo_residual.py's update_actor for the identical recovery/log_prob
        rationale. KL penalty is computed against ref_residual_actor_params
        (the residual's own frozen snapshot), not against the VLA."""
        def loss_fn(params):
            observations = batch_encode(self.batch_encoder.apply_fn, params["batch_encoder"], batch["observations"])
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)

            dist = self.residual_actor.apply_fn(
                {"params": params["residual_actor"]}, observations, actions=batch["base_actions"], p=batch["states"]
            )
            residual_actions = (batch["actions"] - batch["base_actions"]) / self.edit_scale
            safe_residual = jnp.clip(residual_actions, -1.0 + 1e-6, 1.0 - 1e-6)
            log_probs = dist.log_prob(safe_residual)

            log_ratio = jnp.clip(log_probs - batch["old_log_probs"], -20.0, 20.0)
            ratio = jnp.exp(log_ratio)

            adv = batch["advantages"]
            surr1 = ratio * adv
            surr2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_loss = -jnp.minimum(surr1, surr2).mean()

            # KL penalty against the frozen RESIDUAL reference snapshot --
            # same low-variance estimator as GRPOLearner's own (see that
            # file's comment): KL(pi||pi_ref) ~= exp(dlog) - dlog - 1.
            ref_dist = self.residual_actor.apply_fn(
                {"params": self.ref_residual_actor_params}, observations, actions=batch["base_actions"], p=batch["states"]
            )
            ref_log_probs = ref_dist.log_prob(safe_residual)
            log_ratio_ref = ref_log_probs - log_probs
            kl_penalty = (jnp.exp(log_ratio_ref) - log_ratio_ref - 1.0).mean()

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

        params = {"residual_actor": self.residual_actor.params, "batch_encoder": self.batch_encoder.params}
        grads, info = jax.grad(loss_fn, has_aux=True)(params)

        grad_is_finite = jnp.isfinite(optax.global_norm(grads))

        def _apply(_):
            residual_actor = self.residual_actor.apply_gradients(grads=grads["residual_actor"])
            batch_encoder = self.batch_encoder.apply_gradients(grads=grads["batch_encoder"])
            return residual_actor, batch_encoder

        def _skip(_):
            return self.residual_actor, self.batch_encoder

        residual_actor, batch_encoder = jax.lax.cond(grad_is_finite, _apply, _skip, operand=None)
        info["grad_skipped"] = jnp.logical_not(grad_is_finite).astype(jnp.float32)
        info["residual_actor_param_norm"] = optax.global_norm(residual_actor.params)

        return self.replace(residual_actor=residual_actor, batch_encoder=batch_encoder), info

    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        """`utd_ratio` is the number of GRPO epochs (K). `actor_batch` unused --
        interface parity only. `batch` must contain `episode_returns` and be
        grouped by `group_size` -- see module docstring."""
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
        batch["base_actions"] = batch["base_actions"].reshape(batch["actions"].shape)

        encoded_obs = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, batch["observations"], stop_gradient=True)
        dist = self.residual_actor.apply_fn(
            {"params": self.residual_actor.params}, encoded_obs, actions=batch["base_actions"], p=batch["states"]
        )
        residual_actions = (batch["actions"] - batch["base_actions"]) / self.edit_scale
        safe_residual = jnp.clip(residual_actions, -1.0 + 1e-6, 1.0 - 1e-6)
        old_log_probs = dist.log_prob(safe_residual)

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
