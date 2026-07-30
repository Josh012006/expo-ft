"""BatchProcessor: fetches and mixes training batches from replay buffers."""
import jax
import numpy as np

from expo_ft.agents import restore_replay_buffer
from expo_ft.data.replay_buffer import PiReplayBuffer
from expo_ft.utils.train_utils import clear_batch, combine_batches


class BatchProcessor:
    """Builds critic and optional actor batches for one gradient update.

    Modes:
    - online only (offline_ratio=0): all samples from the online replay buffer
    - mixed (0 < offline_ratio < 1): shuffled online + offline critic batches
    - BCLearner (use_dagger_hil_sampling): critic batch from replay; actor from HIL chunks
    - ON-POLICY (on_policy=True, for PPO/GRPO): a contiguous, time-ordered
      rollout of exactly `rollout_length` transitions collected under the
      CURRENT policy, consumed once then discarded. The three modes above all
      draw uniformly at random across the whole buffer history — correct for
      off-policy learners, mathematically invalid for PPO/GRPO (GAE's backward
      recursion needs consecutive timesteps, and the importance ratio
      exp(log_pi - log_pi_old) is only meaningful when pi_old really is the
      policy that collected the actions).
    """
    def __init__(
        self,
        replay_buffer: PiReplayBuffer,
        offline_replay_buffer: PiReplayBuffer,
        data_sharding,
        batch_size: int,
        utd_ratio: int,
        offline_ratio: float,
        actor_success_only: bool,
        use_dagger_hil_sampling: bool,  # True for BCLearner: actor batch from HIL chunks only
        dataset=None,
        on_policy: bool = False,
        rollout_length: int = 0,
        replan_steps: int = 1,
    ):
        if dataset is not None:
            # offline_ratio=0: seed demos into the online replay buffer only.
            if offline_ratio == 0 or use_dagger_hil_sampling:
                replay_buffer.insert_dataset(dataset)
            if offline_ratio != 0:
                offline_replay_buffer.insert_dataset(dataset)

        self.replay_buffer = replay_buffer
        self.offline_replay_buffer = offline_replay_buffer
        self.data_sharding = data_sharding
        self.batch_size = batch_size
        self.offline_ratio = offline_ratio
        self.actor_success_only = actor_success_only
        self.use_dagger_hil_sampling = use_dagger_hil_sampling
        self.on_policy = on_policy
        self.rollout_length = int(rollout_length)
        self.replan_steps = int(replan_steps)
        # Transitions collected since the last update. Reset to 0 on every
        # next_batch() so each update consumes a rollout gathered entirely
        # under the policy in force since the previous update.
        self._rollout_count = 0

        if on_policy:
            if self.rollout_length <= 0:
                raise ValueError("on_policy=True requires rollout_length > 0")
            # Need rollout_length usable starts PLUS replan_steps of lookahead
            # so every sampled index has a valid next_* target that is itself
            # part of this same freshly-collected rollout (rather than stale
            # data from a previous policy, or positions not yet written).
            needed = self.rollout_length + self.replan_steps
            if needed > replay_buffer._capacity:
                raise ValueError(
                    f"rollout_length ({self.rollout_length}) + replan_steps "
                    f"({self.replan_steps}) exceeds replay buffer capacity "
                    f"({replay_buffer._capacity})"
                )
            # Deliberately skip building the random-sampling iterators below:
            # in on-policy mode they'd be both unused and misleading.
            self.replay_iterator = None
            self.offline_iterator = None
            self.hil_iterator = None
            self._ep_buffer_start = replay_buffer._insert_index
            return

        # total_bs MUST end up divisible by utd_ratio (== batch_size exactly,
        # per-utd-step) -- computing online_bs/offline_bs independently via
        # two separate int()/round() truncations does NOT guarantee their sum
        # equals total_bs for an arbitrary offline_ratio (e.g. batch_size=24,
        # utd_ratio=2, offline_ratio=0.7 -> int(48*0.3)=14, int(48*0.7)=33,
        # sum=47 != 48). Only offline_ratio=0.5 happened to divide evenly by
        # luck, which is why this went unnoticed until a different ratio was
        # tried. Fix: round ONE side, derive the other as the complement of
        # the fixed total -- guarantees exact sum regardless of offline_ratio.
        total_bs = batch_size * utd_ratio
        if use_dagger_hil_sampling or offline_ratio == 0:
            online_bs, offline_bs = total_bs, 0
        else:
            offline_bs = round(total_bs * offline_ratio)
            online_bs = total_bs - offline_bs

        self.replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": online_bs,
            },
            data_sharding=data_sharding,
        )

        self.offline_iterator = None
        if offline_ratio > 0 and not use_dagger_hil_sampling:
            self.offline_iterator = offline_replay_buffer.get_iterator(
                sample_args={
                    "batch_size": offline_bs,
                },
                data_sharding=data_sharding,
            )

        self.hil_iterator = None
        if use_dagger_hil_sampling:
            self.hil_iterator = replay_buffer.get_iterator(
                sample_args={"batch_size": batch_size, "hil_only": True},
                data_sharding=data_sharding,
            )

        if actor_success_only and not use_dagger_hil_sampling:
            self._offline_actor_bs = int(batch_size * offline_ratio)
            self._online_actor_bs = batch_size - self._offline_actor_bs

        self._ep_buffer_start = replay_buffer._insert_index

    def insert_transition(self, transition_dict):
        self.replay_buffer.insert(transition_dict)
        if self.on_policy:
            self._rollout_count += 1

    def rollout_ready(self):
        """On-policy only: True once enough fresh transitions have been
        collected under the current policy to form one complete rollout."""
        return self.on_policy and self._rollout_count >= (self.rollout_length + self.replan_steps)

    def on_episode_start(self):
        self._ep_buffer_start = self.replay_buffer._insert_index

    def on_episode_done(self, success):
        if success:
            self.replay_buffer.mark_episode_success(
                self._ep_buffer_start, self.replay_buffer._insert_index
            )
        self._ep_buffer_start = self.replay_buffer._insert_index

    def restore(self, checkpoint_dir, up_to_step=None):
        """Restore replay buffer from disk and rebuild success marks."""
        restore_replay_buffer(checkpoint_dir, self.replay_buffer, up_to_step=up_to_step)
        self.replay_buffer.restore_success_marks()

    def next_batch(self, combine_rng):
        """Return (critic_batch, actor_batch, new_rng) for one update step."""
        if self.on_policy:
            # Take the last (rollout_length + replan_steps) transitions and use
            # the first rollout_length of them as batch positions, so every
            # index's next_* lookup (indices + replan_steps) lands inside this
            # same rollout. Indices are consecutive and in collection order —
            # exactly what GAE's backward recursion assumes.
            capacity = self.replay_buffer._capacity
            end = self.replay_buffer._insert_index
            start = (end - self.rollout_length - self.replan_steps) % capacity
            indices = (start + np.arange(self.rollout_length)) % capacity

            raw = self.replay_buffer.sample_jax(
                batch_size=self.rollout_length, indices=indices
            )
            batch = self.replay_buffer._convert_to_openpi_format(raw)
            batch = self.replay_buffer.apply_data_sharding(batch, self.data_sharding)
            # Consumed: the next update must wait for a rollout gathered under
            # the policy this update is about to produce. This is what makes
            # the data genuinely on-policy rather than merely recent.
            self._rollout_count = 0
            return batch, None, combine_rng

        if self.use_dagger_hil_sampling or self.offline_ratio == 0:
            batch = next(self.replay_iterator)
            new_rng = combine_rng
        else:
            online_batch = next(self.replay_iterator)
            offline_batch = next(self.offline_iterator)
            shuffle_key, new_rng = jax.random.split(combine_rng)
            batch = combine_batches(online_batch, offline_batch, rng=shuffle_key)
            clear_batch(online_batch)
            clear_batch(offline_batch)

        batch = self.replay_buffer.apply_data_sharding(batch, self.data_sharding)

        actor_batch = None
        if self.use_dagger_hil_sampling:
            actor_batch = next(self.hil_iterator)
            actor_batch = self.replay_buffer.apply_data_sharding(actor_batch, self.data_sharding)
        elif self.actor_success_only:
            actor_batch = self._sample_success_actor_batch(new_rng)
            if actor_batch is not None:
                new_rng_parts = jax.random.split(new_rng)
                new_rng = new_rng_parts[0]
                actor_batch = self.replay_buffer.apply_data_sharding(
                    actor_batch, self.data_sharding
                )

        return batch, actor_batch, new_rng

    def _sample_success_actor_batch(self, rng):
        if self.offline_ratio == 0:
            raw = self.replay_buffer.sample_jax(
                self.batch_size, success_only=True
            )
            if raw is not None:
                return self.replay_buffer._convert_to_openpi_format(raw)
            return None

        online_raw = self.replay_buffer.sample_jax(
            self._online_actor_bs, success_only=True
        )
        if online_raw is not None:
            offline_raw = self.offline_replay_buffer.sample_jax(
                self._offline_actor_bs, success_only=True
            )
            if offline_raw is not None:
                online_part = self.replay_buffer._convert_to_openpi_format(online_raw)
                offline_part = self.offline_replay_buffer._convert_to_openpi_format(offline_raw)
                shuffle_key, _ = jax.random.split(rng)
                return combine_batches(online_part, offline_part, rng=shuffle_key)

        raw = self.offline_replay_buffer.sample_jax(
            self.batch_size, success_only=True
        )
        if raw is not None:
            return self.offline_replay_buffer._convert_to_openpi_format(raw)
        return None
