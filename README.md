# ExpoFT — π₀.₅ + ManiSkill RL Fine-Tuning

Sample-efficient RL fine-tuning of π₀.₅ on ManiSkill simulation tasks, using
the ExpoFT algorithm (frozen VLA + trainable residual policy + critic). Started
as a port of the original real-robot ExpoFT algorithm to simulation; has since
grown into a deeper investigation of why RL fine-tuning consistently fails to
beat the SFT baseline (see Current status).

> This repo adapts the original real-DROID-robot ExpoFT codebase
> ([pd-perry/expo-ft](https://github.com/pd-perry/expo-ft)) to run entirely in
> ManiSkill simulation instead — no real robot, NUC, or spacemouse involved.
> `run_pipeline.py` and `expo_ft/env/maniskill_env.py` are new, replacing the
> original client-server DROID environment with a local ManiSkill one.
> `expo_ft/agents/alg/expo_ft_old.py` preserves the original, reference-faithful
> ExpoLearner (MSE scalar critic, REDQ-style ensemble) unmodified, for
> comparison/rollback — `expo_ft/agents/alg/expo_ft.py` (the one actually used)
> has since diverged from it: a categorical/distributional critic architecture
> (XQC/XQCfD-style, see Current status) replaces the original's scalar
> regression critic. Both are directly runnable and A/B-comparable via
> `model_cls: "EXPOLearner"` (new) vs. `"EXPOLearnerOld"` (original) in the
> task YAML — see Pipeline.

## Setup

```bash
git clone --recurse-submodules <this-repo>
cd expo-ft
uv sync
```

`openpi` (`expo_ft/agents/vla/openpi`) and `mani-skill` (`expo_ft/third_party/ManiSkill`)
are git submodules pointing at forks — `--recurse-submodules` is required, or
you'll get empty directories and `uv sync` will fail. If you already cloned
without it:
```bash
git submodule update --init --recursive
```

This installs `openpi`/`openpi-client` and `mani-skill` (both editable, from
the submodule paths above) + all other dependencies.

**If `uv sync` fails to find a package**: check that `pyproject.toml` actually
lists it as a dependency. We've been bitten by this before — `mani-skill` and
`matplotlib` were missing entirely, and `torch`/`imageio`/`gymnasium` were only
resolving as transitive dependencies of something else (fragile). Verify with:
```bash
python -c "import mani_skill.envs, torch, imageio, gymnasium, matplotlib; print('OK')"
```

**For the compute part:**
- Vulkan fix for headless rendering is baked into the job scripts
  (`libvulkan1` download + `VK_ICD_FILENAMES`).
- Use A100L (80GB), not a 40GB A100 — training peaks around 78GB (see the
  `XLA_PYTHON_CLIENT_MEM_FRACTION` note below).
  `job_demos.sh` doesn't need a GPU-heavy card (no model loaded there).
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` is set in `job_rl.sh` — JAX's default
  is only 75% of the card, which was the actual root cause of repeated OOM
  crashes at ~61GB on an 80GB card (see Changelog). Don't remove this.
- On Mila, keep large package caches (`uv`, `openpi`, `huggingface`, `pip`,
  `jax`) on `$SCRATCH` with symlinks back into `~/.cache` — `$HOME` is capped
  at 100GB and these can easily exceed that alone. Keep the repo itself
  (code, `.venv`) on `$HOME`, not `$SCRATCH` — `$SCRATCH` is meant for
  temporary/job files and gets periodically cleaned; only `logs/` and
  `demos/` should be symlinked there given their size.

## Pipeline

Everything runs through `scripts/run_pipeline.py --config <task.yaml> --stage <stage>`:

| Stage | What it does | Job script |
|---|---|---|
| `demos` | Generate + convert demonstrations (motion planning → RGB replay → DROID/LeRobot format) | `job_demos.sh` |
| `sft` | Supervised fine-tuning warmup on demos | `job_sft.sh <venv> <config>` |
| `rl` | ExpoFT RL fine-tuning from an SFT checkpoint — architecture selected by `model_cls` in the task YAML, see below | `job_rl.sh <venv> <config> [sft_checkpoint]` |
| `all` | All of the above in sequence | — |

**Architecture toggle**: `model_cls` in the task YAML picks which critic
architecture `train_pi_robo.py` dispatches to — `EXPOLearner` (categorical
critic, current default) or `EXPOLearnerOld` (original scalar-critic
architecture, for direct A/B comparison — see Current status). Each has its
own model config (`configs/model/{expo_ft,expo_ft_old}_pi_config.py`) and its
own task YAML per task (`configs/task/maniskill/<task>_{sft,expo_ft,expo_ft_old}.yaml`)
— the `_sft.yaml` variant is shared for the `demos`/`sft` stages; the
architecture-specific variants are used for `--stage rl`.
`run_pipeline.py::stage_rl` reads `model_cls` from whichever task YAML is
passed and picks the matching model config automatically.

Evaluation:
- `scripts/eval_policy.py` (single checkpoint) — `--checkpoint <sft_dir>` for an
  SFT checkpoint, or `--rl-checkpoint <rl_checkpoints_dir>/<step>` for a full
  RL/EXPOLearner checkpoint (residual policy + critic included, not just the
  frozen VLA — see Changelog, this needed a real fix). `job_eval.sh <venv>
  <config> <n_episodes> [checkpoint] [rl_checkpoint]`.
- `scripts/eval_curve.py` — sweeps every checkpoint in a directory on a fixed
  set of episode seeds, with ±1 SE error bars. Add `--rl-curve` when sweeping
  RL checkpoints, and `--start-checkpoint <sft_dir>` to use the SFT checkpoint
  an RL run started from as the curve's step-0 reference point (instead of the
  untrained base model, which wouldn't reflect what RL improved upon).
  `--save-videos` writes one subdirectory per checkpoint under a shared
  `videos/` folder. `job_eval_curve.sh <venv> <config> <checkpoints_dir>
  <n_episodes> [save_videos] [start_checkpoint] [rl_curve]`.

Tasks currently in use: **StackCube-v1**, **PushCube-v1**, **PickCube-v1**
(the goal-marker visibility patch — see Known Issues — is required for
PickCube to be usable at all).

## Current status (July 2026)

SFT is fully validated on all three tasks (see Published checkpoints below).
**RL fine-tuning still does not beat the SFT baseline on any task, with either
critic architecture tried so far** — but a lot of
what was an open question in the previous write-up (below, kept for context)
has since been diagnosed, and two real, independent bugs have been found and
fixed along the way without resolving the core symptom.

**Critic architecture**: replaced the original scalar-regression critic
(REDQ-style ensemble, MSE loss against an unbounded TD target) with a
categorical/distributional one (XQC, arXiv 2509.25174 / XQCfD, arXiv
2605.10734 — fixed bounded support instead of a scalar, batch norm + weight
norm on the critic MLP, no ensemble). See
`expo_ft/networks/categorical_value.py` and `expo_ft/agents/alg/expo_ft.py`;
the original architecture is preserved unmodified in `expo_ft_old.py`. Result:
`target_q_max`/`target_q_min` now genuinely converge and stay bounded instead
of climbing indefinitely (the original architecture's `critic_loss` also grew
increasingly spiky over training; the new one stays smooth) — but
`eval/success_rate` still collapses the same way regardless. This was an
important negative result: it rules out critic-training instability itself as
the primary driver of the collapse.

**Reward/done/mask timing bug (found and fixed)**: in `train_pi_robo.py`'s
main loop, `env.get_info_for_step()` (reward/done/mask) was being called
*before* `env.step()` instead of after — so every stored transition received
the reward resulting from the *previous* action instead of the one actually
being stored, a systematic one-step misattribution on every transition
collected online. Also
delayed episode-done detection by one step. Confirmed via a controlled
synthetic reproduction (not just re-reading the code) that this is a real,
structural mismatch between the accumulated n-step reward window and the
observation transition it's supposed to explain — not a labeling artifact
compensated for elsewhere. Scope: only the online RL loop; SFT, demo
generation, and critic pretraining (all separate pipelines) were never
affected. Fixed by reordering the fetch to happen after `env.step()`. A
controlled before/after comparison (same hyperparameters otherwise) showed
`target_q_max` stabilizing similarly to the categorical-critic result above,
but again no change to `eval/success_rate`. Our tasks use ManiSkill's default
dense (`normalized_dense`) reward, not sparse — the shaped component of the
reward is only mildly perturbed by a one-step shift, but the discrete success
bonus layered on top (`reward[success] += ...`) is exactly the kind of
discontinuous value this bug would misattribute most.

**Current leading hypothesis**: the argmax candidate-selection mechanism
itself (`EXPOLearner.sample_batch_actions` — the same critic both picks its
favorite among 16 base+edited candidates *and* evaluates that choice for the
TD bootstrap target — classic maximization/self-reference bias, closely
related to why Double Q-learning exists). Two independent, confirmed fixes
(critic architecture, reward timing) each improved something real and
measurable without touching `eval/success_rate`, which points at this
mechanism by elimination rather than direct proof.

**XQCfD mitigations being tested, in order**:
1. Critic pretraining (BC/TD warm-start on demos before RL starts,
   `rl_critic_pretrain_steps` in the ExpoFT task YAML) — tested, did not help
   on its own.
2. KL regularization against the SFT policy, replacing the generic entropy
   bonus (`rl_kl_coef`/`rl_kl_ref_std`, computed in closed form in pre-tanh
   Gaussian space — see `expo_ft.py`'s `update_residual_actor`). **Note:**
   `rl_kl_coef` is additive alongside the existing entropy bonus in this
   implementation, not a replacement like in the paper (there, one
   coefficient serves both roles) — set `rl_entropy_scale: 0.0` for a faithful
   isolated test. An initial confounded run (KL + entropy both active) showed
   the most encouraging result of this whole investigation so far:
   `eval/success_rate` consistently 5–10 points above an entropy-only
   baseline for most of training. An isolated KL-only test is in progress to
   confirm how much of that is attributable to KL specifically.
3. Stationary/HetStat architecture (not yet implemented) — queued.

**Also queued**: decoupled selection/evaluation for the argmax mechanism
(Double-DQN style — use the *online* critic to select the best candidate,
the *target* critic to evaluate that specific choice, rather than the target
critic doing both) — no new critic needed, since ExpoFT already carries an
online/target pair. Orthogonal to the KL/HetStat changes above (touches
critic usage, not the residual policy's own loss), so it can be layered on
top or tested in isolation at any point without reverting anything.

<details>
<summary>Previous write-up (superseded by the above, kept for history)</summary>

RL fine-tuning currently degrades the SFT policy on every task and every
hyperparameter configuration tried so far — success rate consistently
collapses after an initial stable period, correlated in timing with growing
instability in `training/critic_loss`. This holds even when starting from a
strong SFT checkpoint (86% success) and even when reproducing the original
paper's exact `utd_ratio=20` within the paper's own validated training length
(~20k steps) — that specific test collapsed *faster* than our reduced
`utd_ratio=2` configuration, showing `utd_ratio` itself (not total training
duration) is the dominant driver of the instability observed here.
</details>

See Changelog for the RL-hyperparameter fixes made along the way, and the
`configs/model/expo_ft_pi_config.py` vs. task-YAML hyperparameter comparison
below. This remains an open, unresolved research question at the time of
writing — not a known bug (though several real bugs were found and fixed
while investigating it).

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

## Camera & embodiment configuration (YAML fields)

```yaml
camera_width: 224
camera_height: 224
camera_eye_pos: [0.1, 0.4, 0.4]      # external camera position
camera_target_pos: [0, 0, 0.1]       # what the external camera looks at
camera_fov: 1.0                      # matches the human-render camera's FOV
robot_uids: panda_wristcam           # panda_wristcam adds a wrist-mounted camera
```

Read by `expo_ft/env/maniskill_env.py`, passed to ManiSkill via
`gym.make(..., sensor_configs=..., robot_uids=...)` — a config-only change,
no code edits needed to reposition/resize the external camera or switch robot
embodiment. `scripts/capture_camera_comparison.py --config <task.yaml> [--seed N]`
renders both the sensor and human-render camera views for visual verification
before committing to a change.

Demo generation applies the same overrides via `scripts/replay_trajectory_patched.py`
(see Changelog) — regenerate demos after changing any of these fields.

## RL hyperparameters (YAML fields)

These apply to the ExpoFT task YAMLs (`<task>_expo_ft.yaml` /
`<task>_expo_ft_old.yaml`).

```yaml
rl_lr: 3.0e-4                    # learning rate for critic and actor (NOTE: write scientific
                                  # notation WITH a decimal point — bare "3e-4" parses as a
                                  # string, not a float, in PyYAML; see Changelog)
rl_discount: 0.99                # discount factor gamma for Bellman backup
rl_tau: 0.005                    # polyak averaging coefficient for critic target network
rl_init_temperature: 1.0         # initial SAC entropy temperature
rl_hidden_dims: [256, 256, 256]  # hidden layer sizes for the edit policy MLP
rl_edit_scale: 0.2               # max magnitude of residual action (paper: 0.05–0.2 by task difficulty)
actor_success_only: true         # if true, actor batch is sampled only from successful transitions
utd_ratio: 2                     # gradient updates per new transition collected (paper: 20 — see Current status)
offline_ratio: 0.5               # fraction of batch from a separate offline demo buffer
                                  # (paper actually uses 0 — demos inserted directly into the
                                  # single online buffer; 0.5 is our own deviation, tested as a
                                  # stability lever, not the paper's default)

# Categorical critic (XQC/XQCfD-style, bounded support — see Current status)
rl_num_atoms: 101                # number of fixed support bins
rl_v_min: -10.0                  # lower bound of the fixed support (NORMALIZED reward units)
rl_v_max: 20.0                   # upper bound of the fixed support (NORMALIZED reward units)
rl_reward_scale_decay: 0.99      # EMA decay for the running reward-RMS estimate used to
                                  # normalize rewards before the Bellman projection, so
                                  # v_min/v_max stay meaningful without per-task hand-tuning

# Critic pretraining (XQCfD-style warm-start on demos before RL starts)
rl_critic_pretrain_steps: 0      # 0 = disabled

# KL regularization for the edit policy (XQCfD-style, see Current status for
# the additive-vs-replacement caveat relative to the paper)
rl_kl_coef: 0.0                  # 0.0 = disabled (exact no-op)
rl_kl_ref_std: 1.0               # std of the fixed N(0, ref_std) reference, pre-tanh space
rl_entropy_scale: 1.0            # weight of the (separate, additive) entropy bonus —
                                  # set to 0.0 alongside rl_kl_coef for an isolated KL test
```

These are read directly by `train_pi_robo.py` and explicitly override the
corresponding fields in `configs/model/expo_ft_pi_config.py` — see Changelog
for why this override wiring was needed (these used to be silently ignored).

## Dataset size & resuming (YAML fields)

```yaml
num_demos_generate: 550  # episodes to GENERATE via motion planning (--stage demos, one-time)
num_data_sft: 50     # episodes used for SFT (0 = every episode in the LeRobot dataset)
num_data_rl: 50      # episodes loaded into the RL offline replay buffer (0 = all)
sft_resume_dir: null # resume an existing SFT run from this exact directory
rl_resume_dir: null  # resume an existing RL run from this exact directory
```

`num_demos_generate` is a different concept from `num_data_sft`/`num_data_rl`
above — how many demos to *generate*, vs. how many of the already-generated
demos to *load*. Both demo-count-for-training fields limit an already-converted
dataset to its first N episodes — no reconversion, no config duplication. SFT
checkpoints auto-namespace when `num_data_sft > 0` (e.g. `..._sft_demos50`) so
a limited-demo run never collides with a full-dataset run.

`sft_resume_dir`/`rl_resume_dir` are deliberately separate fields (not a
single shared `resume_dir`) — SFT and RL are different runs with different
directories, and `run_pipeline.py`/`train_pi_robo.py` each resolve their own
run directory independently (see Changelog).

These used to be CLI overrides (`--num-demos` on `run_pipeline.py`); they're
YAML-only now so a run's full configuration lives in one place.

## Changelog — research-phase fixes (July 2026)

Everything below is from the investigation described in Current status —
kept separate from the ManiSkill-adaptation changelog further down since it's
a different phase of work (debugging *why* RL doesn't beat SFT, rather than
getting the ManiSkill port running at all).

**Categorical critic architecture** (`expo_ft/networks/categorical_value.py`,
`expo_ft/agents/alg/expo_ft.py`): replaced the scalar MSE-regression critic
with a C51-style categorical one (fixed bounded support, batch norm + weight
norm, no ensemble) per XQC/XQCfD. `expo_ft_old.py` preserves the original
architecture unmodified for comparison/rollback — a thin passthrough at
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

**`EXPOLearnerOld` toggle**: `expo_ft_old.py` was previously just a passive
fallback file, not actually runnable. Wired it into `train_pi_robo.py`'s
dispatch and `run_pipeline.py`'s model-config lookup as `model_cls:
"EXPOLearnerOld"`, plus a thin `expo_ft_old_pi_config.py` (reuses
`expo_ft_pi_config.py` as-is — `expo_ft_old.create()`'s `**kwargs` silently
absorbs the categorical-critic-specific fields it doesn't need) and
per-task YAMLs, so the original architecture is directly A/B-testable against
the categorical rewrite rather than just preserved as a rollback reference.

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
`--start-checkpoint` support (see Pipeline section above).

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
submodules — see Setup above for the current recommended layout.

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
GitHub account — see Setup for the `--recurse-submodules` requirement.

**Tooling added:** `eval_curve.py` (checkpoint sweeps, fixed episode seeds, SE
error bars), `validate_demos_full_pipeline.py` (rigorous end-to-end demo
replay validation), `capture_camera_comparison.py` (visual camera
verification, supports `--seed`), `diagnose_reward_timing.py` (originally
written to document that the reward/action timing convention matched the
original ExpoFT reference implementation; later revisited and found to be a
real bug regardless — matching the reference doesn't establish correctness,
just provenance — see the research-phase Changelog above for the actual fix).

## Published checkpoints

SFT checkpoints (LoRA, JAX/orbax format — see each model card for why no
PyTorch conversion is provided) are published on HuggingFace:

- [`josh11234/ExpoFT-Pi05-StackCube-v1-SFT`](https://huggingface.co/josh11234/ExpoFT-Pi05-StackCube-v1-SFT) (41% success on 200 held-out seeds)
- [`josh11234/ExpoFT-Pi05-PushCube-v1-SFT-62p`](https://huggingface.co/josh11234/ExpoFT-Pi05-PushCube-v1-SFT-62p) (62% success on 200 held-out seeds)
- [`josh11234/ExpoFT-Pi05-PushCube-v1-SFT-86p`](https://huggingface.co/josh11234/ExpoFT-Pi05-PushCube-v1-SFT-86p) (86% success on 200 held-out seeds)
- [`josh11234/ExpoFT-Pi05-PickCube-v1-SFT`](https://huggingface.co/josh11234/ExpoFT-Pi05-PickCube-v1-SFT) (22% success on 200 held-out seeds)

No RL checkpoints are published — RL has not yet produced a policy that
improves on these SFT baselines (see Current status above).

## Original paper

*"EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for
Vision-Language-Action Models"* — [Project Website](https://pd-perry.github.io/expo-ft) | [arXiv](https://arxiv.org/abs/2605.25477)

```bibtex
@misc{dong2026expoft,
      title={EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models},
      author={Perry Dong and Kuo-Han Hung and Tian Gao and Dorsa Sadigh and Chelsea Finn},
      year={2026},
      eprint={2605.25477},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.25477},
}
```
