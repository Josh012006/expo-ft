"""PPOLearner: on-policy Proximal Policy Optimization, from pixels.

Structurally quite different from EXPOLearner/SACLearner: PPO is on-policy,
so it needs a state-value baseline (not a state-action Q-function), a
Generalized Advantage Estimate (GAE) computed over whole trajectories rather
than random i.i.d. transitions, and a clipped surrogate objective computed
against log-probabilities recorded under the policy that actually collected
the data.

IMPORTANT — this changes what `batch` must contain relative to
EXPOLearner/SACLearner/BCLearner:
  - `batch` must be an genuinely ON-POLICY rollout collected under the CURRENT
    policy since the last `update()` call — not a random sample from an
    off-policy replay buffer. This learner does not maintain or read from a
    replay buffer itself.
  - `batch` must be shaped (num_steps, ...) i.e. time-ordered within each
    trajectory (needed for the backward GAE recursion), with a `masks` field
    that is 0 exactly at true terminations (not truncations) — same
    convention as EXPOLearner's `masks` field.
  - `old_log_probs`/`old_values` are NOT required to be supplied by the
    caller: `update()` computes them itself as its first step, under the
    CURRENT (pre-update) actor/value params, since `batch` is assumed to be
    genuinely on-policy at call time. This keeps PPOLearner self-contained
    without requiring changes to the rollout/replay-buffer code elsewhere in
    this pipeline.

`utd_ratio` is reinterpreted here as the number of PPO epochs (K in the PPO
paper) to run over the given rollout batch — matching the existing pipeline's
calling convention (`agent.update(agent, batch, cfg.utd_ratio, ...)`) without
introducing a new config field.

No residual/edit mechanism, no frozen VLA action-head reliance — same as
SACLearner. The VLA `vla` object is used only for input preprocessing and
output denormalization.
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
from expo_ft.agents.alg.batch_utils import prepare_critic_batch
from expo_ft.data.dataset import DatasetDict
from expo_ft.distributions import TanhNormal
from expo_ft.networks import MLP, BatchEncoder, Ensemble
from expo_ft.networks.pixel_multiplexer import PixelTanhNormalMultiplexer
from expo_ft.networks.state_action_value import StateValue
from expo_ft.networks.encoders import ResNetV2Encoder
from expo_ft.utils.augmentation import make_data_augmentation_fn


def _split_params(agent: Any) -> tuple[Any, dict[str, at.Params]]:
    batch_encoder_params = agent.batch_encoder.params
    actor_params = agent.actor.params
    value_params = agent.value.params

    agent = dataclasses.replace(
        agent,
        batch_encoder=dataclasses.replace(agent.batch_encoder, params={}),
        actor=dataclasses.replace(agent.actor, params={}),
        value=dataclasses.replace(agent.value, params={}),
    )
    params = {
        "batch_encoder_params": batch_encoder_params,
        "actor_params": actor_params,
        "value_params": value_params,
    }
    return agent, params


def _merge_params(agent: Any, params: dict[str, at.Params]) -> Any:
    batch_encoder = dataclasses.replace(agent.batch_encoder, params=params["batch_encoder_params"])
    actor = dataclasses.replace(agent.actor, params=params["actor_params"])
    value = dataclasses.replace(agent.value, params=params["value_params"])
    return dataclasses.replace(agent, batch_encoder=batch_encoder, actor=actor, value=value)


def restore_checkpoint(checkpoint_manager, agent, step: int | None = None):
    agent, params = _split_params(agent)
    restored = checkpoint_manager.restore(step, items={"agent": agent, "params": params})
    return _merge_params(restored["agent"], restored["params"])


def save_checkpoint(checkpoint_manager: ocp.CheckpointManager, agent: Any, step: int):
    agent, params = _split_params(agent)
    checkpoint_manager.save(step, {"agent": agent, "params": params})


def load_agent(seed, example_observation, example_action, example_state,
                actor, actor_train_state, target_actor_params, agent_kwargs, metadata,
                mesh, data_sharding, replicated_sharding, resume, replan_steps,
                default_prompt, **kwargs):
    """Create a PPOLearner. `actor` is the pre-built VLA, used only for input
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
    return PPOLearner.create(seed, example_observation, example_action, example_state, **agent_kwargs)


@partial(jax.jit, static_argnames=("encoder_fn", "stop_gradient"))
def batch_encode(encoder_fn, encoder_params, observations, stop_gradient=False):
    return encoder_fn({"params": encoder_params}, observations, stop_gradient=stop_gradient)


def compute_gae(rewards, values, masks, next_value, discount, gae_lambda):
    """Backward GAE recursion over a (num_steps,) trajectory.

    `masks` must be 0 exactly at true terminations (not at truncations due to
    max_episode_steps) — a truncated-but-not-terminated final step should
    still bootstrap from `next_value`, matching the same convention already
    established for EXPOLearner's Bellman backup.
    """

    def _step(carry, xs):
        gae, next_v = carry
        reward, value, mask = xs
        delta = reward + discount * next_v * mask - value
        gae = delta + discount * gae_lambda * mask * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _step, (jnp.zeros_like(next_value), next_value), (rewards, values, masks), reverse=True
    )
    returns = advantages + values
    return advantages, returns


class PPOLearner(AgentLearner, struct.PyTreeNode):
    """On-policy PPO: stochastic policy + state-value baseline, clipped surrogate objective."""

    rng: jax.random.PRNGKey
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    vla: Any = struct.field(pytree_node=False)
    batch_encoder: TrainState
    actor: TrainState
    value: TrainState
    discount: float
    gae_lambda: float
    clip_eps: float
    value_clip_eps: Optional[float] = struct.field(pytree_node=False)
    value_loss_coef: float
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
        # PPO hyperparameters
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_clip_eps: Optional[float] = 0.2,
        value_loss_coef: float = 0.5,
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
        print("[PPOLearner] observation shape:", observation_space.shape)
        print("[PPOLearner] action shape:", actions.shape, "action horizon:", action_horizon, "action_dim:", action_dim)
        print("[PPOLearner] states shape:", states.shape)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, value_key, encoder_key = jax.random.split(rng, 4)

        encoder_cls = partial(ResNetV2Encoder, stage_sizes=encoder_stage_sizes, num_filters=encoder_num_filters)
        batch_encoder_def = BatchEncoder(
            encoder_cls=encoder_cls, latent_dim=latent_dim_image, pixel_keys=pixel_keys, depth_keys=depth_keys
        )
        batch_encoder_params = batch_encoder_def.init(encoder_key, observation_space)["params"]
        batch_encoder = TrainState.create(
            apply_fn=batch_encoder_def.apply, params=batch_encoder_params, tx=optax.adam(learning_rate=critic_lr)
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

        value_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm)
        # Value network: same pixel+state input wiring as the actor, but a plain
        # state-value head (StateValue) instead of a distribution — PixelMultiplexer
        # (not PixelTanhNormalMultiplexer, which is distribution-specific) fits this.
        from expo_ft.networks import PixelMultiplexer

        value_net_cls = partial(Ensemble, net_cls=partial(StateValue, base_cls=value_base_cls), num=1)
        value_def = PixelMultiplexer(network_cls=value_net_cls, latent_dim=latent_dim_state, include_state=include_state)
        value_params = value_def.init(value_key, critic_observations, p=critic_states)["params"]
        value_tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm) if max_grad_norm is not None else optax.identity(),
            optax.adam(learning_rate=critic_lr),
        )
        value = TrainState.create(apply_fn=value_def.apply, params=value_params, tx=value_tx)
        value_shape = jax.eval_shape(lambda: value)
        value_sharding = _sharding.fsdp_sharding(value_shape, mesh, log=True)
        value = jax.jit(lambda x: x, in_shardings=replicated_sharding, out_shardings=value_sharding)(value)

        agent = cls(
            rng=rng,
            data_augmentation_fn=make_data_augmentation_fn(use_full_augmentation),
            vla=vla,
            batch_encoder=batch_encoder,
            actor=actor,
            value=value,
            discount=discount,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            value_clip_eps=value_clip_eps,
            value_loss_coef=value_loss_coef,
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
        padded = jnp.zeros((1, self.action_horizon, self.vla.model_config.action_dim)).at[:, : self.replan_steps, : self.action_dim].set(action)
        raw_action = self.vla.process_transformed_outputs(padded)[0]
        n = min(self.replan_steps, self.action_horizon)
        action = raw_action[:n].reshape(n, self.action_dim)
        sample_info = {"sample_time": 0.0}
        return jnp.array(action), self.replace(rng=rng), sample_info

    def update_actor(self, batch: DatasetDict) -> Tuple["PPOLearner", Dict[str, float]]:
        """Single-minibatch PPO actor+value update. Expects `batch` to already carry
        `old_log_probs`, `advantages`, `returns` (see `update()`, which computes these
        before calling this per-minibatch)."""
        key, rng = jax.random.split(self.rng)

        def loss_fn(params):
            observations = batch_encode(self.batch_encoder.apply_fn, params["batch_encoder"], batch["observations"])
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)

            dist = self.actor.apply_fn({"params": params["actor"]}, observations, p=batch["states"])
            log_probs = dist.log_prob(batch["actions"])
            ratio = jnp.exp(log_probs - batch["old_log_probs"])

            adv = batch["advantages"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            surr1 = ratio * adv
            surr2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_loss = -jnp.minimum(surr1, surr2).mean()

            values = self.value.apply_fn({"params": params["value"]}, observations, p=batch["states"])
            if self.value_clip_eps is not None:
                clipped_values = batch["old_values"] + jnp.clip(
                    values - batch["old_values"], -self.value_clip_eps, self.value_clip_eps
                )
                value_loss = 0.5 * jnp.maximum(
                    (values - batch["returns"]) ** 2, (clipped_values - batch["returns"]) ** 2
                ).mean()
            else:
                value_loss = 0.5 * ((values - batch["returns"]) ** 2).mean()

            entropy = dist.entropy().mean() if hasattr(dist, "entropy") else -log_probs.mean()
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            approx_kl = (batch["old_log_probs"] - log_probs).mean()
            clip_frac = (jnp.abs(ratio - 1.0) > self.clip_eps).mean().astype(jnp.float32)

            return loss, {
                "ppo_loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                "entropy": entropy, "approx_kl": approx_kl, "clip_frac": clip_frac,
                "ratio_mean": ratio.mean(),
            }

        params = {"actor": self.actor.params, "value": self.value.params, "batch_encoder": self.batch_encoder.params}
        grads, info = jax.grad(loss_fn, has_aux=True)(params)

        actor = self.actor.apply_gradients(grads=grads["actor"])
        value = self.value.apply_gradients(grads=grads["value"])
        batch_encoder = self.batch_encoder.apply_gradients(grads=grads["batch_encoder"])
        info["actor_param_norm"] = optax.global_norm(actor.params)
        info["value_param_norm"] = optax.global_norm(value.params)

        return self.replace(actor=actor, value=value, batch_encoder=batch_encoder, rng=rng), info

    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        """`utd_ratio` is the number of PPO epochs (K) over this on-policy rollout batch.
        `actor_batch` is unused (PPO has no separate success-only actor sampling — accepted
        only for interface parity with the rest of the pipeline)."""
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

        # 1. Compute old_log_probs / old_values / advantages / returns under the CURRENT
        # (pre-update) params — valid since `batch` is assumed to be a fresh on-policy
        # rollout collected under exactly these params.
        encoded_obs = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, batch["observations"], stop_gradient=True)
        dist = self.actor.apply_fn({"params": self.actor.params}, encoded_obs, p=batch["states"])
        old_log_probs = dist.log_prob(batch["actions"])
        old_values = self.value.apply_fn({"params": self.value.params}, encoded_obs, p=batch["states"])

        next_encoded_obs = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, batch["next_observations"], stop_gradient=True)
        next_values = self.value.apply_fn({"params": self.value.params}, next_encoded_obs, p=batch["next_states"])
        # Bootstrap from the LAST step's own next_value (standard GAE convention).
        next_value = next_values[-1]

        advantages, returns = compute_gae(
            batch["rewards"], old_values, batch["masks"], next_value, self.discount, self.gae_lambda
        )

        batch["old_log_probs"] = old_log_probs
        batch["old_values"] = old_values
        batch["advantages"] = advantages
        batch["returns"] = returns

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
