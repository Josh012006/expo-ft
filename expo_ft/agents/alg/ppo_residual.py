"""PPOResidualLearner: on-policy PPO, but trained on a RESIDUAL policy
conditioned on the frozen base VLA's own action, instead of a standalone
actor with zero connection to it (see ppo.py's PPOLearner and its own
module docstring: "No residual/edit mechanism, no frozen VLA action-head
reliance").

Why this exists: pi0.5 is flow-matching-based and exposes no tractable
action log-probability, which is why PPOLearner never conditions on it and
instead trains a small, separately-initialized TanhNormal actor from
scratch (needing its own BC warm-start on demos just to approximate what
the SFT-competent VLA already does natively -- see PPOLearner's
pretrain_actor_bc docstring). This sidesteps that problem a different way:
PPO never needs pi0.5's own log-prob at all here, only the RESIDUAL's --
which, like ExpoFT's own residual actor, is a standard TanhNormal with a
perfectly tractable log-prob. The frozen VLA supplies the base action
(sampled, not trained), and PPO's entire clipped-surrogate machinery
(ratio, GAE, entropy) operates purely on the small residual correction
added on top of it. This is the standard "residual policy learning"
parametrization (e.g. Johannink et al. 2019, Silver et al. 2018), just
applied here with pi0.5 as the frozen base.

Reuses, verbatim in spirit: the residual actor construction (TanhNormal +
PixelEditMultiplexer, conditioned on the base action) from expo_ft.py, and
the GAE/clipped-surrogate/minibatch machinery from ppo.py. No critic-based
argmax-over-candidates selection anywhere in this file -- the residual is
sampled directly, once, like any ordinary stochastic policy.

IMPORTANT -- what `batch` must additionally contain, on top of ppo.py's own
on-policy requirements (see that module's docstring):
  - `base_actions`: the frozen VLA's own action, in ITS OWN normalized
    space (same space sample_training_actions() already returns, and the
    same space the residual's TanhNormal operates in -- no extra
    normalization needed). This is NOT re-sampled during update() -- pi0.5
    is a stochastic flow-matching model, so a fresh sample would not match
    the one actually used to construct `batch["actions"]` during rollout.
    It must be recorded at rollout time (see sample_actions() below, which
    returns it via sample_info["base_action"]) and threaded through into
    the replay buffer (PiReplayBuffer(store_base_actions=True) -- see
    replay_buffer.py) and from there into every training batch.

`utd_ratio` is reinterpreted as the number of PPO epochs (K), matching
PPOLearner's own convention.
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
from expo_ft.agents.alg.ppo import compute_gae, batch_encode
from expo_ft.data.dataset import DatasetDict
from expo_ft.distributions import TanhNormal
from expo_ft.networks import MLP, BatchEncoder, Ensemble
from expo_ft.networks.pixel_multiplexer import PixelEditMultiplexer, PixelMultiplexer
from expo_ft.networks.state_action_value import StateValue
from expo_ft.networks.encoders import ResNetV2Encoder
from expo_ft.utils.augmentation import make_data_augmentation_fn


def _split_params(agent: Any) -> tuple[Any, dict[str, at.Params]]:
    batch_encoder_params = agent.batch_encoder.params
    residual_actor_params = agent.residual_actor.params
    value_params = agent.value.params

    agent = dataclasses.replace(
        agent,
        batch_encoder=dataclasses.replace(agent.batch_encoder, params={}),
        residual_actor=dataclasses.replace(agent.residual_actor, params={}),
        value=dataclasses.replace(agent.value, params={}),
    )
    params = {
        "batch_encoder_params": batch_encoder_params,
        "residual_actor_params": residual_actor_params,
        "value_params": value_params,
    }
    return agent, params


def _merge_params(agent: Any, params: dict[str, at.Params]) -> Any:
    batch_encoder = dataclasses.replace(agent.batch_encoder, params=params["batch_encoder_params"])
    residual_actor = dataclasses.replace(agent.residual_actor, params=params["residual_actor_params"])
    value = dataclasses.replace(agent.value, params=params["value_params"])
    return dataclasses.replace(agent, batch_encoder=batch_encoder, residual_actor=residual_actor, value=value)


_restore_checkpoint, _save_checkpoint = make_checkpoint_fns(_split_params, _merge_params)


def restore_checkpoint(checkpoint_manager, agent, step: int | None = None):
    return _restore_checkpoint(checkpoint_manager, agent, step)


def save_checkpoint(checkpoint_manager: ocp.CheckpointManager, agent: Any, step: int):
    _save_checkpoint(checkpoint_manager, agent, step)


def load_agent(seed, example_observation, example_action, example_state,
                actor, actor_train_state, target_actor_params, agent_kwargs, metadata,
                mesh, data_sharding, replicated_sharding, resume, replan_steps,
                default_prompt, **kwargs):
    """Create a PPOResidualLearner. Unlike PPOLearner, `actor_train_state` IS
    used here (to actually sample the frozen VLA's own base action), not
    just `actor` for pre/post-processing -- see sample_actions()."""
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
    return PPOResidualLearner.create(seed, example_observation, example_action, example_state, **agent_kwargs)


class PPOResidualLearner(AgentLearner, struct.PyTreeNode):
    """On-policy PPO trained on a residual policy conditioned on the frozen
    base VLA's own sampled action. See module docstring."""

    rng: jax.random.PRNGKey
    data_augmentation_fn: Callable = struct.field(pytree_node=False)
    vla: Any = struct.field(pytree_node=False)
    actor_train_state: Any = struct.field(pytree_node=False)  # the frozen VLA's own train_state, used only to sample base actions -- never updated
    batch_encoder: TrainState
    residual_actor: TrainState
    value: TrainState
    edit_scale: float = struct.field(pytree_node=False)
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
        actor_train_state: Any = None,
        action_horizon: int = 1,
        mesh: Optional[Any] = None,
        # PPO hyperparameters (identical defaults to PPOLearner, for comparability)
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_clip_eps: Optional[float] = 0.2,
        value_loss_coef: float = 0.5,
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
        print("[PPOResidualLearner] observation shape:", observation_space.shape)
        print("[PPOResidualLearner] action shape:", actions.shape, "action horizon:", action_horizon, "action_dim:", action_dim)
        print("[PPOResidualLearner] states shape:", states.shape, "edit_scale:", edit_scale)

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
        critic_actions = jnp.expand_dims(actions, axis=0)

        # Residual actor: SAME construction as ExpoFT's own (expo_ft.py),
        # conditioned on the base action via PixelEditMultiplexer (which
        # concatenates `actions` -- the base action -- into the input
        # features before the TanhNormal network). No critic-based
        # candidate selection anywhere here: this distribution is sampled
        # directly, once, exactly like any ordinary on-policy actor.
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

        value_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm)
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
            actor_train_state=actor_train_state,
            batch_encoder=batch_encoder,
            residual_actor=residual_actor,
            value=value,
            edit_scale=edit_scale,
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
            "residual_actor_params": jax.device_put(self.residual_actor.params, s),
        })

    def sample_actions(self, observations, only_base_actions=False):
        """Samples base_action ~ frozen VLA, residual ~ TanhNormal(base_action),
        executes base_action + edit_scale * residual. `only_base_actions`
        (interface parity with EXPOLearner) skips the residual entirely and
        executes the raw base action -- useful as a sanity check that the
        base action alone behaves like plain SFT rollout would.

        Returns (action, new_agent, sample_info) where sample_info now
        additionally carries "base_action" (in the residual's own
        normalized space) -- this MUST be threaded through into the replay
        buffer (see this module's docstring) for update() to later recompute
        the residual's log_prob against the exact base action used here,
        not a fresh (and generally different) resample.
        """
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

        # Sample the frozen VLA's own base action -- NOT trained, NOT
        # updated by any gradient here; actor_train_state is used purely as
        # a sampling source, exactly as ExpoFT's own sample_batch_actions
        # uses self.actor/self.actor_train_state for the same purpose.
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

    def update_actor(self, batch: DatasetDict) -> Tuple["PPOResidualLearner", Dict[str, float]]:
        """Single-minibatch PPO update. `batch["actions"]` is the COMBINED
        (base + edit_scale*residual) action that was actually executed;
        `batch["base_actions"]` is the base action it was built from.
        Recovers the residual that was actually sampled by inverting the
        combination, and computes the log-prob of THAT residual under the
        residual actor's own (re-evaluated, current-params) distribution --
        never pi0.5's own log-prob, which is never needed here."""
        key, rng = jax.random.split(self.rng)

        def loss_fn(params):
            observations = batch_encode(self.batch_encoder.apply_fn, params["batch_encoder"], batch["observations"])
            observations = jax.lax.with_sharding_constraint(observations, self.data_sharding)

            dist = self.residual_actor.apply_fn(
                {"params": params["residual_actor"]}, observations, actions=batch["base_actions"], p=batch["states"]
            )
            # Recover the residual actually sampled at rollout time:
            # actions = base_actions + edit_scale * residual.
            residual_actions = (batch["actions"] - batch["base_actions"]) / self.edit_scale
            # Same tanh-boundary safety clip as ppo.py's loss_fn -- see that
            # comment. Clips the RESIDUAL (what log_prob is actually taken
            # of), not the combined action.
            safe_residual = jnp.clip(residual_actions, -1.0 + 1e-6, 1.0 - 1e-6)
            log_probs = dist.log_prob(safe_residual)

            log_ratio = jnp.clip(log_probs - batch["old_log_probs"], -20.0, 20.0)
            ratio = jnp.exp(log_ratio)

            adv = batch["advantages"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            surr1 = ratio * adv
            surr2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_loss = -jnp.minimum(surr1, surr2).mean()

            values = self.value.apply_fn({"params": params["value"]}, observations, p=batch["states"])[0]
            if self.value_clip_eps is not None:
                clipped_values = batch["old_values"] + jnp.clip(
                    values - batch["old_values"], -self.value_clip_eps, self.value_clip_eps
                )
                value_loss = 0.5 * jnp.maximum(
                    (values - batch["returns"]) ** 2, (clipped_values - batch["returns"]) ** 2
                ).mean()
            else:
                value_loss = 0.5 * ((values - batch["returns"]) ** 2).mean()

            # See ppo.py's matching comment: TanhNormal.entropy() raises
            # NotImplementedError (Tanh-squashed Gaussian has no closed form);
            # fall back to the sample-based proxy in that case.
            try:
                entropy = dist.entropy().mean()
            except NotImplementedError:
                entropy = -log_probs.mean()
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            approx_kl = (batch["old_log_probs"] - log_probs).mean()
            clip_frac = (jnp.abs(ratio - 1.0) > self.clip_eps).mean().astype(jnp.float32)

            return loss, {
                "ppo_loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                "entropy": entropy, "approx_kl": approx_kl, "clip_frac": clip_frac,
                "ratio_mean": ratio.mean(),
            }

        params = {"residual_actor": self.residual_actor.params, "value": self.value.params, "batch_encoder": self.batch_encoder.params}
        grads, info = jax.grad(loss_fn, has_aux=True)(params)

        # Same NaN-guard as PPOLearner's own update_actor -- see that
        # file's comment for why this matters more here (num_epochs x
        # jax.lax.scan carrying the agent forward).
        grad_is_finite = jnp.isfinite(optax.global_norm(grads))

        def _apply(_):
            residual_actor = self.residual_actor.apply_gradients(grads=grads["residual_actor"])
            value = self.value.apply_gradients(grads=grads["value"])
            batch_encoder = self.batch_encoder.apply_gradients(grads=grads["batch_encoder"])
            return residual_actor, value, batch_encoder

        def _skip(_):
            return self.residual_actor, self.value, self.batch_encoder

        residual_actor, value, batch_encoder = jax.lax.cond(grad_is_finite, _apply, _skip, operand=None)
        info["grad_skipped"] = jnp.logical_not(grad_is_finite).astype(jnp.float32)
        info["residual_actor_param_norm"] = optax.global_norm(residual_actor.params)
        info["value_param_norm"] = optax.global_norm(value.params)

        return self.replace(residual_actor=residual_actor, value=value, batch_encoder=batch_encoder, rng=rng), info

    def update(self, agent, batch: DatasetDict, utd_ratio: int, actor_batch: DatasetDict = None):
        """`utd_ratio` is the number of PPO epochs (K) over this on-policy rollout batch.
        `actor_batch` is unused -- accepted only for interface parity with the pipeline."""
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
        # prepare_critic_batch only knows about "actions" -- base_actions
        # needs the same un-chunking/reshape treatment applied to it
        # separately, matching whatever shape "actions" ends up in.
        batch["base_actions"] = batch["base_actions"].reshape(batch["actions"].shape)

        encoded_obs = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, batch["observations"], stop_gradient=True)
        dist = self.residual_actor.apply_fn(
            {"params": self.residual_actor.params}, encoded_obs, actions=batch["base_actions"], p=batch["states"]
        )
        residual_actions = (batch["actions"] - batch["base_actions"]) / self.edit_scale
        safe_residual = jnp.clip(residual_actions, -1.0 + 1e-6, 1.0 - 1e-6)
        old_log_probs = dist.log_prob(safe_residual)
        old_values = self.value.apply_fn({"params": self.value.params}, encoded_obs, p=batch["states"])[0]

        next_encoded_obs = batch_encode(self.batch_encoder.apply_fn, self.batch_encoder.params, batch["next_observations"], stop_gradient=True)
        next_values = self.value.apply_fn({"params": self.value.params}, next_encoded_obs, p=batch["next_states"])[0]
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
