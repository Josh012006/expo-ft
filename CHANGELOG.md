# ExpoFT — Detailed Status & Changelog

Full project history, detailed findings, and known issues for the ExpoFT
π₀.₅ + ManiSkill RL fine-tuning work — moved out of the main README to keep
that one short. See [README.md](README.md) for setup, pipeline usage, and
the configuration reference (camera/embodiment, RL hyperparameters, dataset
size & resuming).

## Current status (September 2026)

**StackCube now also beats its SFT baseline.** `edit_scale=0.2`, extended to
2500 episodes total, reaches **0.755 success rate on the deterministic,
200-episode `eval_rigorous` protocol** — up from a ~0.40 SFT baseline. The
blocker that had kept StackCube (and by implication PickCube, not yet
re-tested) behind PushCube for so long turned out to be unrelated to
`edit_scale`, reward shaping, or the argmax-selection bias already
investigated above: it was `replan_steps=8`, a value StackCube's task YAML
had carried since the very start of the project, never revisited once
PushCube's own YAML was independently changed to `replan_steps=4` on July 15
(same commit that fixed the reward/done/mask timing bug — the two changes
were unrelated but landed together, which delayed noticing the difference).

**A collaborator's max-entropy SAC hypothesis, tested directly and mostly
ruled out as the cause — but a genuine, precisely-characterized structural
finding came out of testing it.** The hypothesis: the learned SAC temperature
(`training/temperature`, α — multiplies the entropy term
`entropy_scale · α · log π(a|s)` in the residual actor's loss, see RL
hyperparameters below) reacts too slowly, leaving the residual policy
effectively random. Tested by setting `rl_init_temperature` two orders of
magnitude below default (`0.01` vs. `1.0`) on an otherwise-matched StackCube
run: `training/entropy` and the newly-added `training/residual_std_mean` (see
below) were statistically indistinguishable from the `1.0` baseline over
110K+ steps. Root cause: the residual policy's distribution is
`tanh(N(μ, σ²))` (see `expo_ft/distributions/tanh_normal.py`), and the
*differential entropy* of a Gaussian squashed through `tanh` is **not**
monotonic in σ — it has a single interior maximum, verified numerically at
**σ* ≈ 0.8749** (per action dimension), matching the observed
`residual_std_mean` (~0.87–0.88) to 3 significant figures, identically across
every task and `edit_scale` tested. At this fixed point the entropy term's
own gradient is ≈0, so scaling it by any α — large or small — moves nothing;
this is why the temperature-magnitude test showed no effect. Confirmed this
is genuinely a σ-only phenomenon: `residual_mean_norm` (the same pre-tanh
distribution's mean) is **not** pinned this way — it's shaped normally by
the Q term, and is exactly where the real, task-dependent divergence turned
out to live (next paragraph). One follow-up test worth recording as a clean
negative result: pushing `rl_init_temperature` a further 10× lower (`0.001`)
*did* move both `training/entropy` and `training/residual_std_mean` away
from this fixed point — genuinely, not a measurement artifact — but
destructively: `eval_rigorous/success_rate` collapsed to 0 on that run.
Evidently the fixed point has finite, not infinite, restoring strength, and
escaping it early in training (before the critic is well-calibrated) is
destabilizing rather than helpful.

**The real StackCube-specific divergence: the residual mean, not the std.**
`residual_mean_norm` (pre-tanh Gaussian mean, ‖μ‖₂) is, like std, genuinely
shaped by the Q term rather than pinned to a fixed point — but unlike std it
diverged sharply by task/config: under StackCube's long-standing
`replan_steps=8`, it shrank continuously toward zero over 100K+ steps with
no sign of stabilizing (the residual policy learning, increasingly, to leave
the frozen base VLA's action untouched); under PushCube's `replan_steps=4`
(unchanged since July 15) it stabilized at a healthy, non-zero plateau
(~0.25–0.3) from early training onward. Re-running StackCube at
`replan_steps=4` reproduced PushCube's stable-plateau pattern almost
exactly, and its `eval_rigorous/success_rate` overtook the `replan_steps=8`
run's in absolute terms (not just faster from a lower starting point) by
~100K steps. **Caveat, stated plainly because it matters for how far this
generalizes**: PushCube has never actually been tested at `replan_steps=8` —
its SFT/demos and every prior run have used `replan_steps=4` since mid-July,
so "the same fix would have been needed for PushCube too" is untested, not
confirmed. What's confirmed is narrower: `replan_steps=8` is the identified
cause of StackCube's specific collapse pattern, and switching to 4 resolves
it.

A partial mechanistic lead, not yet fully confirmed: `replan_steps` sets
`full_action_dim` (`= replan_steps × action_dim`) for both the RL agent *and*
the SFT/BC actor loss (`expo_ft/agents/alg/batch_utils.py::prepare_critic_batch`
truncates each demo's full action chunk to its first `replan_steps` actions
for supervision) — meaning a checkpoint's own SFT training implicitly assumes
a matching eval-time `replan_steps`. StackCube's published SFT checkpoint was
trained under `replan_steps=8`; evaluating/fine-tuning it under
`replan_steps=4` is a genuine train/eval consistency change, though exactly
how that translates into the specific mean-collapse pattern observed isn't
nailed down — the fix works empirically, the full causal chain is still
open.

Two ablations at `replan_steps=4`, both clean negative results worth keeping
on record so they aren't re-tried blind:
- `edit_scale=10.0`: `mean_d_actions_norm` explodes to 25–30 by 15K steps and
  stays there, `target_q_mean` never leaves ≈0, `eval_rigorous/success_rate`
  collapses toward 0 by 20K steps — the same large-`edit_scale` failure mode
  documented in the July research-phase changelog below, confirmed to persist
  under the new `replan_steps` value (i.e. not a `replan_steps`-specific
  interaction).
- `rl_init_temperature=0.001`: see the max-entropy paragraph above.

**New instrumentation, purely additive (no change to any loss/gradient
computation)**, added to `expo_ft/agents/alg/expo_ft.py`'s
`residual_actor_loss_fn` — ported from `expo_ft_categorical.py`, which
already had it from the earlier KL-regularization work:
- `training/residual_std_mean`, `training/residual_mean_norm` — pre-tanh
  Gaussian std / mean-norm, see above for what each revealed.
- `rl_temp_lr` (YAML field) — the temperature optimizer's own learning rate
  was previously silently decoupled from `rl_lr`, always pinned at
  `sac_config.py`'s `3e-4` default regardless of what `rl_lr` was set to in
  the task YAML (no override wiring existed, unlike `actor_lr`/`critic_lr`).
  Now independently tunable; kept at `3.0e-4` (unchanged behavior) in every
  task YAML until deliberately changed.

**A new diagnostic to separate the residual's own contribution from the
critic's candidate-selection effect — added, not yet run/analyzed.** Even
"deterministic" `eval_rigorous` retains a source of real improvement
unrelated to the residual: `sample_actions()`'s own docstring confirms the
frozen base VLA still draws N *stochastic* flow-matching samples regardless
of `deterministic=True` (only the residual's own sampling is made
deterministic), and the trained critic argmax-selects among them. So a small
or shrinking residual mean does not imply "only the untouched base VLA is
running" — it can equally mean "critic-based best-of-N selection over the
frozen VLA's own stochastic samples is doing real work, independent of the
residual." Added `n_edit_samples=0` as an eval-time override
(`agent.replace(n_edit_samples=0)` — cheap, since it's a static/non-pytree
struct field; doesn't touch any trained weights) that keeps the critic's
selection active but skips the residual entirely, isolating that
contribution. Available two ways:
- Live during training: `eval_rigorous/success_rate_critic_only` (+
  `_stderr`), logged at the same interval and on the same fixed episode seeds
  as the normal `eval_rigorous/success_rate`.
- Post-hoc, for already-finished runs: `scripts/eval_curve_critic_only.py`
  (`job_eval_curve_critic_only.sh <venv> <config> <checkpoints_dir>
  <n_episodes> [start_checkpoint]`) — sweeps every checkpoint, running both
  variants on the same fixed seed list, producing `comparison.json` /
  `comparison.png` (two curves: full pipeline vs. critic-only).

**Currently in progress**: speeding up StackCube's convergence *rate*, not
just its eventual success rate (2500 episodes to reach 0.755 is slow) —
testing `replan_steps`, `batch_size`, and `utd_ratio` as the main levers,
having already ruled out `rl_init_temperature` tuning (see above) as a
productive direction on its own.

<details>
<summary>Current status (August 2026) — superseded by the above</summary>

**PushCube now beats its SFT baseline** — the first task on which RL
fine-tuning has produced a policy that improves on the frozen starting point,
using the original (MSE/REDQ) critic architecture, a corrected sparse reward
signal, and a much higher update-to-data ratio than initially tested (see
below). StackCube and PickCube do not yet match this, but the most likely
reason has been identified (a stale, overly small residual-action budget left
over from earlier testing — see below) and both are being re-run with it
corrected.

**The reward signal was wrong the whole time the previous write-up below was
current.** A collaborator flagged implausible-looking critic value estimates,
which led to finding that ManiSkill's `reward_mode` was never explicitly set
at environment creation, silently defaulting to a continuous, shaped
(`normalized_dense`) reward instead of the sparse, binary one this whole
algorithm (and the original paper) assumes. Confirmed directly by replaying a
real demonstration and checking that essentially none of its logged reward
values were exactly zero, where a working sparse signal should be zero almost
everywhere. A second, independent bug compounded this specifically for the
categorical critic architecture: its reward normalization divided by a
running estimate of the reward's own scale, which shrank as success became
rarer during training, inflating the effective reward for the few remaining
successes — a self-reinforcing loop, confirmed structurally absent from the
original MSE/REDQ architecture. `reward_mode: "sparse"` is now set explicitly
in every task YAML (with a code-level `getattr(cfg, "reward_mode", "sparse")`
diagnostic metric used to detect it (`misrank_rate`) measurably improved
(from saturating at 0.92–1.00 under the old dense reward, to 0.72–0.87 under
sparse) but did not fully disappear — consistent with the bias being a real,
partial contributor rather than the sole explanation for the earlier
degradation. Critically: **PushCube's successful run uses no special
mitigation against this bias at all** — no decoupled candidate selection, no
frozen critic encoder, no critic pretraining, standard Polyak target update —
meaning the reward-mode fix and update-ratio tuning alone were sufficient to
recover strong performance on this task, without any intervention on the
critic or target-network mechanism itself.

**StackCube and PickCube's lagging results traced to a stale config value,
not (yet) a deeper problem.** Both tasks' YAMLs still had `rl_edit_scale:
0.05` — a leftover from an earlier, superseded round of testing — never
updated when the same field was corrected to `0.2` for PushCube. Given both
tasks' SFT baselines are meaningfully weaker than PushCube's, a residual
policy with only a 0.05 budget likely could not meaningfully correct the
weaker base behavior. Both are being re-run with the corrected value.

**Update-to-data ratio (`utd_ratio`) matters a lot, and pushing it higher
surfaced a separate engineering problem.** Higher `utd_ratio` produced better
results, but resuming a run from a checkpoint at `utd_ratio` above ~20
reliably crashed with an out-of-memory error, while starting the same
configuration fresh never did. This turned into a substantial debugging
effort, described in full in the Changelog below — summary: a real upstream
bug in JAX's handling of `jax.lax.scan` combined with automatic
rematerialization (confirmed against a public bug report matching this
project's exact pinned JAX version, `jax==0.5.3`, itself pinned by `openpi`
and not something this project can freely change) was the original trigger;
working around it by replacing the scan with a Python-level loop then
introduced its own, separate memory-donation bug. Both are fixed. Checkpoint
resuming is now expected to be reliable at high `utd_ratio`.

**A rigorous, deterministic evaluation protocol is now built directly into
training**, per a collaborator's suggestion that the existing in-training
rolling-window proxy (`eval/success_rate`) is a useful but insufficient
substitute for a proper held-out measurement. At fixed, regularly-spaced
step intervals (`rl_eval_interval`), including once before any training at
all, the policy is evaluated on `rl_eval_episodes` fixed-seed episodes with
the residual policy's stochastic sampling replaced by its deterministic mode
— logged separately as `eval_rigorous/success_rate` (+ standard error) so it
is never confused with the rolling-window proxy. See RL hyperparameters
below and Changelog for the full design (including a subtle early bug where
the very first, "step 0" evaluation point wasn't actually measuring the
clean frozen baseline it was supposed to).

**GPU utilization stays well under full accelerator usage throughout
training**, investigated but not resolved. The training loop is
fundamentally sequential — the accelerator sits idle while each simulated
environment step is computed on CPU. The most direct fix (running multiple
environments in parallel) is architecturally unavailable: ManiSkill's
`physx_cpu` backend hard-disallows `num_envs > 1` (this project's own fork
confirms this isn't just undocumented — ManiSkill's own changelog lists a bug
fix for a case where `physx_cpu` used to incorrectly *permit* `num_envs > 1`).
Switching to `physx_cuda`, the backend that does support it, carries a
documented risk (from ManiSkill's own docs) of subtly different simulated
physics from what this project's demonstrations and SFT were generated
under — judged too risky to introduce without re-validating the whole data
pipeline. Separately confirmed that `pi0.5`'s own real-world DROID
pretraining used a 15Hz control frequency, while ManiSkill's default (used
unmodified by all three tasks here) is 20Hz — a real, confirmed mismatch,
documented here as a known limitation rather than fixed, since correcting it
would mean regenerating demonstrations and repeating SFT from scratch. Note
this is unrelated to this project's own `control_hz` YAML field, which is
purely a wall-clock pacing throttle in the training loop and has no effect on
simulated physics or action semantics either way.

<details>
<summary>Previous write-up (superseded by the above, kept for history)</summary>

RL fine-tuning still does not beat the SFT baseline on any task, with either
critic architecture tried so far — but a lot of what was an open question in
the write-up before *that* (further below) has since been diagnosed, and two
real, independent bugs have been found and fixed along the way without
resolving the core symptom.

**Critic architecture**: replaced the original scalar-regression critic
(REDQ-style ensemble, MSE loss against an unbounded TD target) with a
categorical/distributional one (XQC, arXiv 2509.25174 / XQCfD, arXiv
2605.10734 — fixed bounded support instead of a scalar, batch norm + weight
norm on the critic MLP, no ensemble). Result: `target_q_max`/`target_q_min`
now genuinely converge and stay bounded instead of climbing indefinitely, but
`eval/success_rate` still collapses the same way regardless — ruling out
critic-training instability itself as the primary driver.

**Reward/done/mask timing bug (found and fixed)**: `env.get_info_for_step()`
was being called before `env.step()` instead of after, so every stored
transition received the reward from the *previous* action. Fixed by
reordering. No change to `eval/success_rate` either.

**Leading hypothesis at the time**: the argmax candidate-selection mechanism
itself (the same critic both picks its favorite candidate and evaluates that
choice for the TD bootstrap target). Several literature-based mitigations
(critic pretraining, KL regularization, decoupled selection) were tested
against it — see the August 2026 Changelog below for how this played out once
the reward-mode bug (above) was also found and fixed.

</details>

</details>


## Known issues / open items

- **Camera setup mismatch with π₀.₅-DROID's training distribution** — fixed:
  external camera repositioned to an actual side view (`camera_eye_pos`/
  `camera_target_pos`), FOV matched to the human-render camera (`camera_fov`,
  was defaulting to a wider FOV than intended, making the same position look
  more zoomed-out than expected), and PushCube's missing wrist camera fixed
  via `robot_uids: panda_wristcam` (root cause: PushCube/PickCube default to
  plain `"panda"`, StackCube already used `"panda_wristcam"` — this is the
  actual difference between the Panda v2/v3 URDFs, not a scene/config issue).
- **Resolution** — fixed: `camera_width`/`camera_height: 224` renders natively
  at the model's input resolution instead of upsampling from 128.
- **PickCube-v1** — goal marker visibility fixed via a monkeypatch
  (`expo_ft/env/patches.py`, since ManiSkill hides it from sensor cameras by
  default). Confirmed working and PickCube-v1 is back in the active task set
  (all RL-stage experiments now cover all three tasks).
- **Control-frequency mismatch with π₀.₅-DROID's training distribution** —
  confirmed, **not fixed**. `pi0.5`'s own real-world DROID pretraining used
  15Hz; ManiSkill's default `control_freq` (used unmodified by all three
  tasks) is 20Hz. Fixing this means regenerating demonstrations and repeating
  SFT from scratch — deprioritized accordingly. Not to be confused with this
  project's own `control_hz` YAML field, which is a wall-clock pacing
  throttle in `train_pi_robo.py`'s loop only, never reaches ManiSkill at all,
  and does not address this.
- **GPU utilization stays low, and the most direct fix is unavailable** —
  `physx_cpu` (required for this project's demo-conversion pipeline, and the
  backend all demos/SFT were generated under) hard-disallows `num_envs > 1`
  in ManiSkill, so true environment parallelism isn't possible without
  switching to `physx_cuda`, which carries a documented risk of subtly
  different simulated physics from what the demonstrations were generated
  with. See the August 2026 Changelog for the full investigation.


## Changelog — research-phase fixes (July 2026)

Everything below is from the investigation described in Current status —
kept separate from the ManiSkill-adaptation changelog further down since it's
a different phase of work (debugging *why* RL doesn't beat SFT, rather than
getting the ManiSkill port running at all).

**Categorical critic architecture** (`expo_ft/networks/categorical_value.py`,
`expo_ft/agents/alg/expo_ft_categorical.py`): replaced the scalar MSE-regression critic
with a C51-style categorical one (fixed bounded support, batch norm + weight
norm, no ensemble) per XQC/XQCfD. `expo_ft.py` preserves the original
architecture for comparison/rollback (and is the current default again — see
Current status) — a thin passthrough at
`expo_ft.py`'s old location was used during the transition so the rest of the
package (`__init__.py`, which every other learner's import chain went
through) didn't hard-depend on whichever architecture was mid-rewrite.
`checkpoint_utils.py` was factored out (generic `restore_checkpoint`/
`save_checkpoint` mechanics, parametrized by each learner's own
`_split_params`/`_merge_params`) so this and future architecture swaps
wouldn't risk breaking BC's own checkpointing, which shared the
same code before this.

**Adaptive reward normalization**: `v_min`/`v_max` apply to *normalized*
reward units — rewards are divided by a running RMS estimate
(`reward_scale_decay`) before the Bellman projection, so the fixed support
stays meaningful across tasks without per-task hand-tuning of the bounds
themselves. Verified via a synthetic test that a sudden 50× jump in a task's
reward scale doesn't break the boundedness guarantee — the normalization
absorbs it.

**Reward/done/mask timing bug** — see Current status for the finding itself;
`get_info_for_step()` moved from before to after `env.step()` in
`train_pi_robo.py`'s main loop.

**Critic pretraining** (`rl_critic_pretrain_steps`): runs
`update_critic()` — unmodified, same argmax mechanism — repeatedly on
offline-only batches before the main training loop starts, to test XQCfD's
"critic/actor coherence" warm-start idea in isolation from everything else.
Logged under `pretrain/*` with `wandb.define_metric` giving it its own step
axis (`pretrain_step`), so it doesn't collide with the main loop's step
counter — an earlier version tried a negative-step convention on the shared
axis instead, which silently dropped every logged point once wandb's
background system-metrics logging (GPU utilization etc., independent of
anything in this code) had already advanced the shared counter past 0.

**KL regularization** (`rl_kl_coef`/`rl_kl_ref_std`) — see Current status.
Computed in closed form in the pre-tanh Gaussian space (`dist.distribution`,
the same attribute already used by `TanhTransformedDistribution.mode()`) —
not the squashed action space, which has no clean closed-form KL between two
Tanh-transformed distributions, the same underlying reason TFP can't compute
`.entropy()` for one either.

**Wandb negative-axis logging bug**: an earlier draft of the critic
pretraining feature logged its metrics on a negative step range (`-N..-1`)
sharing the main loop's default step axis, intending it to render as a
warm-up phase preceding step 0. In practice, wandb's background system
telemetry can advance its shared step counter past 0 before any of this
code's own `wandb.log()` calls run, so every negative-step point got silently
rejected ("steps must be monotonically increasing"). Fixed by giving
`pretrain/*` (and later `actor_pretrain/*`) their own independent step axis
via `wandb.define_metric(..., step_metric=...)`, decoupled from the main
loop's default counter entirely.

**`EXPOLearnerCategorical` toggle**: `expo_ft_categorical.py` was previously just a passive
fallback file, not actually runnable. Wired it into `train_pi_robo.py`'s
dispatch and `run_pipeline.py`'s model-config lookup as `model_cls:
"EXPOLearnerCategorical"`, plus a thin `expo_ft_categorical_pi_config.py` (reuses
`expo_ft_pi_config.py` as-is — `expo_ft_categorical.create()`'s `**kwargs` silently
absorbs the MSE/REDQ-specific fields it doesn't need) and
per-task YAMLs, so the categorical architecture is directly A/B-testable against
the categorical rewrite rather than just preserved as a rollback reference.

## Changelog — memory, resuming, and evaluation (August 2026)

**Checkpoint-resume OOM at high `utd_ratio`, root-caused across several
layers.** Resuming at `utd_ratio` above ~20 reliably crashed with a
`RESOURCE_EXHAUSTED` error; fresh runs at the same configuration never did.
In order of discovery:
1. The *original* crash (before any fix below) was a hard XLA compiler
   crash (`Check failed: return_shape->IsTuple()`, not a normal OOM),
   traced to a confirmed upstream JAX bug (`jax-ml/jax#27748`) where
   automatic rematerialization becomes ineffective specifically on
   `jax.remat`-wrapped `jax.lax.scan` — matching this project's exact
   pinned JAX version (`0.5.3`, itself pinned by `openpi`'s own
   `pyproject.toml`, not something this project can change). Fixed by
   restructuring `EXPOLearner.update()`'s `utd_ratio`-many critic updates
   from one `jax.lax.scan` into three separately-JIT'd functions
   (`_prepare_minibatches_jit`/`_critic_update_step_jit`/
   `_update_finalize_jit`) orchestrated by a plain Python loop —
   mathematically identical (same sequential carry, same RNG consumption
   order), just compiled differently.
2. That fix introduced its own, separate memory bug: none of the three
   split functions had `donate_argnames`, so each of the `utd_ratio`
   sequential Python-loop calls allocated a fresh full copy of the agent
   state instead of reusing memory in place — the exact buffer-reuse
   `jax.lax.scan` provided for free via its loop-carried state. This, not
   the original scan bug, turned out to be what made even *fresh* runs
   start crashing at previously-working settings once `utd_ratio`/
   candidate-count went high enough. Fixed by adding
   `donate_argnames=("agent",)` to all three split functions.
3. Donation then surfaced a subtler correctness bug: code that reused the
   real, persistent `agent` object as input to a discarded/throwaway JIT
   call (e.g. a compile-warmup pass) could have its buffers silently freed
   by that donation, corrupting the real agent for the rest of training
   (`RuntimeError: Array has been deleted`). Fixed by always passing an
   explicit `.copy()` of every array leaf into any such throwaway call.
4. Several smaller, independent contributors were also found and fixed:
   checkpoint restore not passing explicit `restore_args` (falls back to a
   slower, more memory-costly "read sharding from file" path); the
   `use_success_batch` static-bool argument compiling a second, separately
   resident program the first time a successful episode appeared post-resume
   (fixed, then reverted after finding it was never actually the proximate
   cause of any of the crashes above); async checkpoint saves not being
   confirmed complete (`checkpoint_manager.wait_until_finished()`) before
   training continued.

**Online replay buffer was never actually surviving a resume, silently.**
Even with all of the above fixed, `training/success_rate` was observed
declining for a stretch after every resume. Root cause: `checkpoint_buffer`
existed as a YAML field but its disk-saving call
(`save_replay_buffer_transition`) was only ever wired into the main training
loop — meaning every resume silently restarted the online buffer from
empty regardless of the flag, forcing training to rebuild data diversity
from scratch each time. Fixed by confirming `checkpoint_buffer: true`
actually engages end-to-end, and by additionally reconstructing
`eval/success_rate`'s own rolling window from the restored buffer's
`dones`/`is_success` history on resume (previously reset to empty
unconditionally too, meaning this metric silently reported nothing for up
to ~20K steps after every resume even once the buffer itself was fixed).

**The dedicated step-0 baseline pass (see next section) had the same
resume gap, closed the same way.** It collects `success_rate_window`
fixed-seed episodes before real training starts; a preemption mid-pass had
no way to resume short of redoing the whole thing, even with
`checkpoint_buffer=true`, since that flag's saving logic didn't cover this
phase either. Fixed by saving this phase's own transitions too (offset
step numbers — `10**9 + step`, chosen because `restore_replay_buffer`'s own
file filter, `str.isdigit()`, is `False` for a leading `-`, so a
negative-offset scheme would have been silently excluded on restore) and
detecting/resuming from an interrupted previous attempt at this pass
specifically, without needing a full agent checkpoint at all — this phase
never updates the agent's own weights, so there is nothing to restore about
the agent itself, only which of the fixed-seed episodes were already
collected.

**Rigorous, deterministic evaluation, integrated directly into training.**
Per a collaborator's suggestion that `eval/success_rate` (a rolling window
over stochastic *online* episodes) is a useful proxy but not a substitute
for a proper held-out measurement: at fixed step intervals (including once
before any training), the residual policy's stochastic sampling is replaced
by its deterministic mode (`TanhTransformedDistribution.mode()`, i.e.
`tanh(mean)` — the frozen base VLA's own flow-matching sampling stays
stochastic either way, it has no equivalent closed-form mode) and evaluated
on a fixed, cached set of episode seeds reused across every evaluation
point. Logged as `eval_rigorous/success_rate` (+ standard error), on its own
`eval_env` instance kept fully separate from the training environment.
Found and fixed along the way: the very first ("step 0") evaluation point
needs `only_base_actions=True` specifically — without it, "step 0" measures
the frozen SFT VLA plus an arbitrary, untrained, randomly-initialized
residual/critic contribution, not the clean baseline every later comparison
is implicitly measured against. The same fix was needed for the
pre-existing `eval/success_rate` initialization pass, which had the
identical issue; the two were then consolidated into one shared pass
(instead of two separate 200-episode rollouts) once both were measuring
the same thing. `scripts/eval_curve.py`'s `main()` was separately found to
be completely non-functional (parsed its CLI arguments but never called
any of `discover_checkpoints`/`run_one_eval`/`rebuild_curve`) and was
rebuilt, restoring `--rl-curve`/`--start-checkpoint`/`--deterministic`.

**GPU utilization investigated, not resolved.** Stays well under full
accelerator usage throughout training. Root cause: the training loop is
fundamentally sequential (VLA inference on GPU, then one simulated
environment step on CPU, repeated), with no overlap between the two. The
most direct fix — running multiple environments in parallel — is
architecturally unavailable under `physx_cpu` (ManiSkill's own changelog
lists a bug fix for a case where this backend used to incorrectly *permit*
`num_envs > 1`; it's a hard `num_envs=1` lock, not just an unsupported
combination). `physx_cuda` does support it, but ManiSkill's own
documentation warns that CPU and GPU-parallelized physics backends are not
guaranteed to produce identical simulated results, particularly for
precision-sensitive tasks — since this project's demonstrations and SFT
were generated entirely on `physx_cpu`, switching the live training
environment to `physx_cuda` risks a silent distribution mismatch between
what SFT learned from and what RL would train against, on the same order of
risk as the original DROID action-space mismatch this project already had
to diagnose once. Judged not worth the risk without first re-validating the
whole data pipeline under the new backend. Separately confirmed (via
`openpi`'s own docs) that `pi0.5`'s real-world DROID pretraining used a
15Hz control frequency, while ManiSkill's own default `control_freq`
(unmodified by any of this project's three tasks) is 20Hz — a real,
confirmed mismatch, documented here as a known limitation rather than
fixed, since correcting it means regenerating demonstrations and repeating
SFT. This project's own `control_hz` YAML field is unrelated to either of
the above: it is a pure wall-clock pacing throttle inside
`train_pi_robo.py`'s own loop, never passed to ManiSkill's `gym.make()` at
all, so changing it is always safe but also does not address the
`control_freq` mismatch just described.

## Changelog — key fixes made while adapting to ManiSkill (July 2026)

**RL checkpoint evaluation was silently impossible before this fix:**
`eval_policy.py` had only ever been built/tested against SFT/openpi-style
checkpoints (`--checkpoint`, loaded via `pi05_weight_loader_path`). Trying to
point it at an RL/EXPOLearner checkpoint crashed with `KeyError: 'params'` —
an RL checkpoint's `"params"` orbax item is a multi-component dict (VLA +
residual actor + critic + temperature + batch encoder params, see
`expo_ft.agents.alg.expo_ft._split_params`), not the simple `{"params": <tree>}`
shape openpi's weight loader expects. Even if that had been fixed, evaluation
would have still silently run with `only_base_actions=True` — evaluating just
the frozen VLA, never the trained residual policy. New `--rl-checkpoint` flag
restores the full agent via orbax's own `restore_checkpoint()` and evaluates
with `only_base_actions=False` so the residual policy + critic-based action
selection actually run. `eval_curve.py` gained matching `--rl-curve` /
`--start-checkpoint` support (see README.md's Pipeline section).

**RL hyperparameters were silently ignored from the task YAML:** `rl_lr`,
`rl_discount`, `rl_tau`, `rl_init_temperature` (previously misnamed
`rl_alpha`), `rl_hidden_dims`, and `rl_edit_scale` (previously
`rl_edit_action_scale`) were all defined in the task YAMLs but never actually
read anywhere in `train_pi_robo.py` — the real values always came from
`configs/model/expo_ft_pi_config.py`'s defaults instead, which happened to
already match the paper for most of these (so no past run was actually
mis-configured by this — but the YAML gave false confidence of control, and
would have silently no-opped if anyone had tried to change one of these
values). Fixed by explicitly wiring `FLAGS.config.X = getattr(cfg, "rl_X",
...)` overrides near the top of `train_pi_robo.py::main()`, executed before
`build_pi05()` reads `FLAGS.config` — verified this ordering is correct
(`build_pi05_config()` does `agent_kwargs = dict(config)`, capturing whatever
mutations were made up to that point).

**PyYAML scientific-notation gotcha:** bare scientific notation without a
decimal point (e.g. `3e-4`) parses as a **string**, not a float — PyYAML
requires `3.0e-4`. This crashed a job the first time `rl_lr` was actually
wired up to be read. All task YAMLs fixed to use the decimal-point form, and
the override code in `train_pi_robo.py` now also defensively wraps every
numeric override in `float(...)` as a second line of defense.

**`num_demos` (in `stage_demos`, controls how many raw demos to *generate*)
was an orphaned field** — no YAML ever defined a field by that name (only
`num_data_sft`/`num_data_rl`, a different concept: how many *already-generated*
demos to load), so it always silently fell back to a hardcoded `550`. Renamed
to `num_demos_generate` and added to all three task YAMLs.

**TensorBoard was silently missing most training metrics** (`critic_loss`,
`actor_loss`, `residual_actor_loss` — only `eval/success_rate` and
`training/loop_time_ms` showed up) because the logging code filtered on
`isinstance(v, (int, float))`, which excludes JAX scalar arrays
(`jnp.float32`). wandb showed everything fine since it accepts JAX arrays
directly. Fixed with an explicit `float()` cast before `tb_writer.add_scalar`.

**Repo migrated from living on `$SCRATCH` to living on `$HOME`** (only
`logs/`/`demos/` remain symlinked to `$SCRATCH`), and `openpi`/`ManiSkill`
converted from untracked/pip-installed dependencies to proper editable git
submodules — see README.md's Setup section for the current recommended layout.

**Pre-SFT pipeline:**
- Fixed `eval_policy.py` unconditionally overriding the DROID-official
  `AssetsConfig` (norm_stats) with local paths, even for baseline eval.
- Fixed a leftover EEF-derived action rescale in `convert_maniskill_to_droid.py`/
  `convert_maniskill_to_lerobot.py` (from an abandoned `pd_ee_delta_pose`
  pivot) that was saturating ~30% of joint-space actions.
- Fixed `max_episode_steps` (env truncation) vs. `max_steps_per_episode` (eval
  loop's own cap) being desynced (100 vs 120).
- Switched `sim_backend` to `physx_cpu` everywhere (control-mode conversion
  requires it; `num_envs=1` is hardcoded anyway so no parallelism lost).

**RL stage (`train_pi_robo.py` / `run_pipeline.py::stage_rl`):**
- `stage_rl` was passing ~15 CLI flags that `train_pi_robo.py` never defines
  in this adaptation (seed/max_steps/batch_size/etc. are read directly from
  the task YAML instead) — stripped down to only the flags actually consumed
  (`--config`, `--task_config`, `--fsdp_devices`, plus
  `--config.<field>=` ml_collections overrides).
- Same norm_stats override bug as `eval_policy.py`, present here too — fixed
  the same way.
- `overwrite=False` was hardcoded in the checkpoint-dir initialization, but
  `main()` always pre-creates the directory first — every fresh run crashed
  with `FileExistsError`. Fixed to `overwrite=not resuming`.
- `actor_success_only` mismatch: `BatchProcessor` correctly reads it from the
  task YAML, but the `EXPOLearner` agent read a separate, unsynced copy from
  the model config (hardcoded `True` there) — causing a crash
  (`NoneType.copy()`) whenever the YAML said `False`. Added an explicit sync
  in `main()`, plus a graceful fallback in `expo_ft.py` for when
  `actor_success_only=True` but no successful episode exists yet in the buffer
  (early in training, or a from-scratch/no-SFT run).
- Added `--sft-checkpoint` to explicitly set which SFT checkpoint RL
  initializes from — previously there was no way to do this, and RL would
  silently fall back to the base pretrained checkpoint.
- `max_to_keep`/`checkpoint_interval` are now configurable via the task YAML
  (checkpoints are ~18GB each — previous defaults filled disk quota fast with
  multiple parallel runs).

**Camera/embodiment overrides also needed in demo generation:** demo RGB
conversion (`replay_trajectory`) runs in ManiSkill's own subprocess with its
own `gym.make(...)` call, completely independent of `maniskill_env.py` —
so camera/resolution/robot_uids overrides silently never reached it, and demos
kept being generated at the old 128×128/no-wristcam settings despite YAML
changes. `scripts/replay_trajectory_patched.py` now also monkeypatches
`gym.make` itself (via a new `--expo-config` arg pointing to the task YAML) to
inject the same `sensor_configs`/`robot_uids` overrides `maniskill_env.py`
uses, so demo generation and eval/RL are guaranteed consistent.

**RL OOM after ~2000+ steps, root-caused (not just worked around):**
`EXPOLearner._update_jit` is `jax.jit`-compiled with `actor_batch` as a
non-static argument. Our own `actor_success_only` cold-start fallback (above)
passed `actor_batch=None` until the first successful episode landed in the
buffer, then switched to passing a real dict — a different pytree structure
each time, which forces JAX to trace and compile (and keep resident) a
*second* XLA program the first time that switch happens, potentially well
into training. Fixed by always passing a consistently-shaped `actor_batch`
(falling back to reusing the main critic `batch`'s own structure) and
controlling the actual branch with a separate `static_argnames` boolean
instead — bounds JAX to exactly the 2 compilations the logic actually needs,
rather than an unplanned structural transition triggered by training dynamics.

**Dataset size and resume directories moved fully into the YAML:**
`--num-demos` (CLI) is gone; replaced by `num_data_sft`/`num_data_rl` fields
so a run's configuration lives in one place instead of being split between
the YAML and job-launch arguments. Likewise the single shared `resume_dir`
(ambiguous between the SFT and RL runs it could refer to) is now
`sft_resume_dir`/`rl_resume_dir` — `resolve_run_dir()` takes the resume
directory as an explicit argument rather than reading a fixed `cfg.resume_dir`
field, so each stage passes its own.

**ManiSkill packaging:** switched from a pinned PyPI install to an editable
install of a fork (`expo_ft/third_party/ManiSkill`, added to `[tool.uv.sources]`
as a `path`+`editable` source — same pattern as `openpi`, deliberately *not*
a `[tool.uv.workspace]` member since ManiSkill's `setup.py`-based packaging
lacks the `[project]` table `uv` workspace membership requires). Lets us track
task/environment modifications as real commits instead of runtime monkeypatches,
and add custom tasks directly.

**Submodule tracking fixed:** `openpi` was listed in `.gitignore` and never
tracked by git at all (silently — no warning, since git ignores it entirely);
`mani-skill`'s fork was a nested git repo `git add -A` couldn't handle either
(the "you've added another git repository" warning). Anyone cloning the repo
before this fix would have gotten empty directories and a broken `uv sync`.
Both are now proper `git submodule`s pointing at forks under the `Josh012006`
GitHub account — see README.md's Setup section for the `--recurse-submodules` requirement.

**Tooling added:** `eval_curve.py` (checkpoint sweeps, fixed episode seeds, SE
error bars), `validate_demos_full_pipeline.py` (rigorous end-to-end demo
replay validation), `capture_camera_comparison.py` (visual camera
verification, supports `--seed`), `diagnose_reward_timing.py` (originally
written to document that the reward/action timing convention matched the
original ExpoFT reference implementation; later revisited and found to be a
real bug regardless — matching the reference doesn't establish correctness,
just provenance — see the research-phase Changelog above for the actual fix).
