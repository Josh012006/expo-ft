"""SACLearner: standard from-pixels Soft Actor-Critic (no VLA action-head reliance).

Unlike EXPOLearner, this learner does NOT treat the pretrained VLA's own
flow-matching action head as a frozen prior to be corrected by a residual
policy. Instead it learns a single, direct policy over the full action space
from scratch, using the same REDQ-style critic ensemble and auto-tuned
temperature as EXPOLearner.

Why this exists: it isolates whether EXPOLearner's specific "residual over a
frozen VLA prior" architecture is itself contributing to the training
instability we've observed, or whether a conventional SAC agent behaves
similarly given the same pixel observations, reward, and replay buffer. It is
also the natural agent to use for an SFT-warmup-necessity ablation (train
directly with no residual structure at all, no frozen VLA prior).

The VLA `vla` object is still used for input preprocessing
(`process_raw_inputs` / `process_transformed_outputs` — image resizing and
action normalization/denormalization against the same DROID norm stats used
throughout this pipeline) but its own `sample_actions` (flow-matching action
head) is never called. All action production goes through this learner's own
policy network.
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
from expo_ft.networks.temperature import Temperature
from expo_ft.data.dataset import DatasetDict
from expo_ft.distributions import TanhNormal
from expo_ft.networks import (
    MLP,
    Ensemble,
    StateActionValue,
    subsample_image_ensemble,
    PixelMultiplexer,
    BatchEncoder,
)
from expo_ft.networks.pixel_multiplexer import PixelTanhNormalMultiplexer
from expo_ft.networks.encoders import ResNetV2Encoder
from expo_ft.utils.augmentation import make_data_augmentation_fn


def _split_params(agent: Any) -> tuple[Any, dict[str, at.Params]]:
    """Split params for checkpointing (actor/critic/temp/encoder — no VLA action head to split)."""
    batch_encoder_params = agent.batch_encoder.params
    actor_params = agent.actor.params
    temp_params = agent.temp.params
    critic_params = agent.critic.params

    agent = dataclasses.replace(
        agent,
        batch_encoder=dataclasses.replace(agent.batch_encoder, params={}),
        actor=dataclasses.replace(agent.actor, params={}),
        temp=dataclasses.replace(agent.temp, params={}),
        critic=dataclasses.replace(agent.critic, params={}),
    )
    params = {
        "batch_encoder_params": batch_encoder_params,
        "actor_params": actor_params,
        "temp_params": temp_params,
        "critic_params": critic_params,
    }
    return agent, params


def _merge_params(agent: Any, params: dict[str, at.Params]) -> Any:
    batch_encoder = dataclasses.replace(agent.batch_encoder, params=params["batch_encoder_params"])
    actor = dataclasses.replace(agent.actor, params=params["actor_params"])
    temp = dataclasses.replace(agent.temp, params=params["temp_params"])
    critic = dataclasses.replace(agent.critic, params=params["critic_params"])
    return dataclasses.replace(agent, batch_encoder=batch_encoder, actor=actor, temp=temp, critic=critic)


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
    """Create a SACLearner. `actor` is the pre-built VLA, used only for input
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
    return SACLearner.create(seed, example_observation, example_action, example_state, **agent_kwargs)


@partial(jax.jit, static_argnames=("critic_fn", "num_min_qs"))
def compute_q(critic_fn, critic_params, observations, actions, states, num_min_qs=None):
    q_values = critic_fn({"params": critic_params}, observations, actions, p=states, sample_num=num_min_qs)
    return q_values.min(axis=0)


@partial(jax.jit, static_argnames=("encoder_fn", "stop_gradient"))
def batch_encode(encoder_fn, encoder_params, observations, stop_gradient=False):
    return encoder_fn({"params": encoder_params}, observations, stop_gradient=stop_gradient)


class SACLearner(AgentLearner, struct.PyTreeNode):
    """Standard from-pixels SAC: single direct policy, REDQ-style critic ensemble, auto-tuned temperature."""

    rng: jax.random.PRNGKey
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    vla: Any = struct.field(pytree_node=False)  # pre-built VLA, used only for input (de)normalization
    batch_encoder: TrainState
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    entropy_scale: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)  # M in RedQ, https://arxiv.org/abs/2101.05982
    action_dim: int = struct.field(pytree_node=False)
    state_dim: int = struct.field(pytree_node=False)
    full_action_dim: int = struct.field(pytree_node=False)
    replan_steps: int = struct.field(pytree_node=False)
    action_horizon: int = struct.field(pytree_node=False)
    resize_size: Optional[int] = struct.field(pytree_node=False)
    default_prompt: Optional[str] = struct.field(pytree_node=False)
    data_sharding: Optional[jax.sharding.NamedSharding] = struct.field(pytree_node=False)
    actor_success_only: bool = struct.field(pytree_node=False)
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
        # SAC hyperparameters
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 10,
        num_min_qs: Optional[int] = 2,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        entropy_scale: float = 1.0,
        init_temperature: float = 1.0,
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
        actor_success_only: bool = False,
        use_full_augmentation: bool = True,
        **kwargs,
    ):
        action_dim = action_space.shape[-1]
        state_dim = states.shape[-1]
        full_action_dim = replan_steps * action_dim
        actions = jnp.zeros((full_action_dim,))
        print("[SACLearner] observation shape:", observation_space.shape)
        print("[SACLearner] action shape:", actions.shape, "action horizon:", action_horizon, "action_dim:", action_dim)
        print("[SACLearner] states shape:", states.shape)

        if target_entropy is None:
            target_entropy = -full_action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key, encoder_key = jax.random.split(rng, 5)

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
        critic_actions = jnp.expand_dims(actions, axis=0)
        critic_states = jnp.expand_dims(states, axis=0)

        # Direct policy over the full action space — no base action to edit, unlike EXPOLearner's residual actor.
        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm)
        actor_dist_cls = TanhNormal(actor_base_cls, full_action_dim)
        actor_def = PixelTanhNormalMultiplexer(
            network_cls=actor_dist_cls, latent_dim=latent_dim_image, include_state=include_state,
            state_latent_dim=latent_dim_state,
        )
        actor_params = actor_def.init(actor_key, critic_observations, p=critic_states)["params"]
        actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optax.adam(learning_rate=actor_lr))
        actor_shape = jax.eval_shape(lambda: actor)
        actor_sharding = _sharding.fsdp_sharding(actor_shape, mesh, log=True)
        actor = jax.jit(lambda x: x, in_shardings=replicated_sharding, out_shardings=actor_sharding)(actor)

        critic_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True, dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm, use_pnorm=use_pnorm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
        critic_def = PixelMultiplexer(network_cls=critic_cls, latent_dim=latent_dim_state, include_state=include_state)
        critic_params = critic_def.init(critic_key, critic_observations, critic_actions, p=critic_states)["params"]
        tx = (
            optax.adamw(learning_rate=critic_lr, weight_decay=critic_weight_decay,
                        mask=lambda p: jax.tree_util.tree_map(lambda _: True, p))
            if critic_weight_decay is not None else optax.adam(learning_rate=critic_lr)
        )
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=tx)
        critic_shape = jax.eval_shape(lambda: critic)
        critic_sharding = _sharding.fsdp_sharding(critic_shape, mesh, log=True)
        critic = jax.jit(lambda x: x, in_shardings=replicated_sharding, out_shardings=critic_sharding)(critic)

        target_critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(apply_fn=temp_def.apply, params=temp_params, tx=optax.adam(learning_rate=temp_lr))
        temp_shape = jax.eval_shape(lambda: temp)
        temp_sharding = _sharding.fsdp_sharding(temp_shape, mesh, log=True)
        temp = jax.jit(lambda x: x, in_shardings=replicated_sharding, out_shardings=temp_sharding)(temp)

        agent = cls(
            rng=rng,
            data_augmentation_fn=make_data_augmentation_fn(use_full_augmentation),
            vla=vla,
            batch_encoder=batch_encoder,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            tau=tau,
            discount=discount,
            target_entropy=target_entropy,
            entropy_scale=entropy_scale,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            action_dim=action_dim,
            state_dim=state_dim,
            full_action_dim=full_action_dim,
            replan_steps=replan_steps,
            action_horizon=action_horizon,
            resize_size=resize_size,
            default_prompt=default_prompt,
            data_sharding=data_sharding,
            actor_success_only=actor_success_only,
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
        """Sample an action from this learner's own policy. `only_base_actions` is accepted
        for interface parity with EXPOLearner/BCLearner but has no effect here — there is no
        separate "base" VLA action to fall back to; this learner always uses its own policy."""
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

        # De-normalize through the VLA's own output pipeline so actions land in the
        # same physical units as EXPOLearner/BCLearner produce (keeps eval/env code shared).
        action = action.reshape(1, self.replan_steps, self.action_dim)
        padded = jnp.zeros((1, self.action_horizon, self.vla.model_config.action_dim)).at[:, : self.replan_steps, : self.action_dim].set(action)
        raw_action = self.vla.process_transformed_outputs(padded)[0]
        n = min(self.replan_steps, self.action_horizon)
        action = raw_action[:n].reshape(n, self.action_dim)
        sample_info = {"sample_time": 0.0}
        return jnp.array(action), self.replace(rng=rng), sample_info

    def update_critic(self, batch: DatasetDict) -> Tuple["SACLearner", Dict[str, float]]:
        rng = self.rng
        key, rng = jax.random.split(rng)

        next_encoded = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, batch["next_observations"], stop_gradient=True)
        next_encoded = jax.device_put(next_encoded, self.data_sharding)
        next_dist = self.actor.apply_fn({"params": self.actor.params}, next_encoded, p=batch["next_states"])
        key, rng = jax.random.split(rng)
        next_actions = next_dist.sample(seed=key)
        next_log_probs = next_dist.log_prob(next_actions)

        key, rng = jax.random.split(rng)
        target_params = subsample_image_ensemble(key, self.target_critic.params, self.num_min_qs, self.num_qs)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params}, next_encoded, next_actions, False, p=batch["next_states"], sample_num=self.num_min_qs,
        )
        next_q = next_qs.min(axis=0) - self.temp.apply_fn({"params": self.temp.params}) * next_log_probs
        target_q = batch["rewards"] + (self.discount ** self.replan_steps) * batch["masks"] * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(params_dict):
            observations = batch_encode(self.batch_encoder.apply_fn, params_dict["batch_encoder"], batch["observations"])
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)
            qs = self.critic.apply_fn(
                {"params": params_dict["critic"]}, observations, batch["actions"], True, p=batch["states"], rngs={"dropout": key},
            )
            critic_loss = (((qs - target_q) ** 2) * batch["valids"]).mean()
            return critic_loss, {
                "critic_loss": critic_loss, "q": qs.mean(), "q_min": qs.min(), "q_max": qs.max(),
                "target_q_mean": target_q.mean(),
            }

        params_dict = {"critic": self.critic.params, "batch_encoder": self.batch_encoder.params}
        grads, info = jax.grad(critic_loss_fn, has_aux=True)(params_dict)
        critic = self.critic.apply_gradients(grads=grads["critic"])
        batch_encoder = self.batch_encoder.apply_gradients(grads=grads["batch_encoder"])
        info["critic_param_norm"] = optax.global_norm(critic.params)

        target_critic_params = optax.incremental_update(critic.params, self.target_critic.params, self.tau)
        target_critic = self.target_critic.replace(params=target_critic_params)
        info["target_critic_param_norm"] = optax.global_norm(target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, batch_encoder=batch_encoder, rng=rng), info

    def update_actor(self, batch: DatasetDict) -> Tuple["SACLearner", Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params):
            observations = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, batch["observations"], stop_gradient=True)
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)
            dist = self.actor.apply_fn({"params": actor_params}, observations, p=batch["states"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn({"params": self.critic.params}, observations, actions, True, p=batch["states"], rngs={"dropout": key2})
            q = qs.mean(axis=0)
            actor_loss = (self.entropy_scale * log_probs * self.temp.apply_fn({"params": self.temp.params}) - q).mean()
            return actor_loss, {"q": q.mean(), "actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> Tuple["SACLearner", Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {"temperature": temperature, "temperature_loss": temp_loss}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)
        return self.replace(temp=temp), temp_info

    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        new_agent, info = self.replace(_infer_cache=None)._update_jit(
            agent.replace(_infer_cache=None), batch, utd_ratio, actor_batch,
        )
        return new_agent.cache_infer_params(), info

    @partial(jax.jit, static_argnames=("utd_ratio",))
    def _update_jit(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        batch = batch.copy()
        rng, key1 = jax.random.split(agent.rng)
        rng, key2 = jax.random.split(rng)
        batch["image"] = self.data_augmentation_fn(key1, batch["image"])
        batch["next_image"] = self.data_augmentation_fn(key2, batch["next_image"])
        batch = prepare_critic_batch(batch, self.vla.model_config.action_dim, self.action_dim, self.state_dim, self.action_horizon, self.replan_steps)
        new_agent = agent.replace(rng=rng)

        total_bs = batch["actions"].shape[0]
        assert total_bs % utd_ratio == 0, f"Batch size ({total_bs}) must be a multiple of utd_ratio ({utd_ratio})"
        minibatch_size = total_bs // utd_ratio

        def reshape_minibatch(x):
            return x.reshape((utd_ratio, minibatch_size) + x.shape[1:])

        minibatches = jax.tree_util.tree_map(reshape_minibatch, batch)

        def create_minibatch_sharding(x):
            ndim = len(x.shape)
            spec_tuple = (None,) + (_sharding.DATA_AXIS,) + (None,) * (ndim - 2)
            new_sharding = jax.sharding.NamedSharding(self.data_sharding.mesh, jax.sharding.PartitionSpec(*spec_tuple))
            return jax.device_put(x, new_sharding)

        minibatches = jax.tree_util.tree_map(create_minibatch_sharding, minibatches)

        def critic_update_step(carry, mb):
            (agent,) = carry
            agent, info = agent.update_critic(mb)
            return (agent,), info

        (new_agent,), critic_infos = jax.lax.scan(critic_update_step, (new_agent,), minibatches)

        last_minibatch = jax.tree_util.tree_map(lambda x: x[-1] if x is not None and hasattr(x, "shape") else x, minibatches)
        new_agent, actor_info = new_agent.update_actor(last_minibatch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        critic_info = jax.tree_util.tree_map(lambda x: x[-1], critic_infos)
        return new_agent, {**actor_info, **temp_info, **critic_info}
