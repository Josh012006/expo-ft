# ExpoFT — π₀.₅ + ManiSkill RL Fine-Tuning

Sample-efficient RL fine-tuning of π₀.₅ on ManiSkill simulation tasks, using
the ExpoFT algorithm (frozen VLA + trainable residual policy + critic). Started
as a port of the original real-robot ExpoFT algorithm to simulation; along the
way turned into a deeper investigation of why RL fine-tuning was failing to
beat the SFT baseline. PushCube now does (see Current status) — StackCube and
PickCube are catching up.

> This repo adapts the original real-DROID-robot ExpoFT codebase
> ([pd-perry/expo-ft](https://github.com/pd-perry/expo-ft)) to run entirely in
> ManiSkill simulation instead — no real robot, NUC, or spacemouse involved.
> `run_pipeline.py` and `expo_ft/env/maniskill_env.py` are new, replacing the
> original client-server DROID environment with a local ManiSkill one.
> `expo_ft/agents/alg/expo_ft_categorical.py` preserves the categorical/
> distributional critic rewrite (XQC/XQCfD-style, C51-bounded support) that
> was briefly the default — `expo_ft/agents/alg/expo_ft.py` (the one actually
> used) is the original, reference-faithful ExpoLearner (MSE scalar critic,
> REDQ-style ensemble), which moved back to being the default once it proved
> more stable under the corrected sparse-reward setup (see Current status).
> Both are directly runnable and A/B-comparable via
> `model_cls: "EXPOLearner"` (MSE, default) vs. `"EXPOLearnerCategorical"`
> (categorical rewrite) in the task YAML — see Pipeline.

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
architecture `train_pi_robo.py` dispatches to — `EXPOLearner` (MSE scalar
critic, current default — see Current status) or `EXPOLearnerCategorical`
(categorical/distributional critic architecture, for direct A/B comparison).
Each has its own model config (`configs/model/{expo_ft,expo_ft_categorical}_pi_config.py`)
and its own task YAML per task (`configs/task/maniskill/<task>_{sft,expo_ft,expo_ft_categorical}.yaml`)
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

## Current status (August 2026)

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
fallback so a future task config that forgets to set it can't silently
reintroduce this), and `use_reward_normalization` lets the affected
normalization be bypassed for the categorical architecture under sparse
reward.

**The maximization-bias mechanism described in the previous write-up is real
but was not the whole story.** Under the corrected sparse reward, the
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
`<task>_expo_ft_categorical.yaml`).

```yaml
rl_lr: 3.0e-4                    # learning rate for critic and actor (NOTE: write scientific
                                  # notation WITH a decimal point — bare "3e-4" parses as a
                                  # string, not a float, in PyYAML; see Changelog)
rl_discount: 0.99                # discount factor gamma for Bellman backup
rl_tau: 0.005                    # polyak averaging coefficient for critic target network
rl_init_temperature: 1.0         # initial SAC entropy temperature
rl_hidden_dims: [256, 256, 256]  # hidden layer sizes for the edit policy MLP
rl_edit_scale: 0.2               # max magnitude of residual action (paper: 0.05–0.2 by task
                                  # difficulty — double check this per-task in the YAML you're
                                  # actually using; a stale 0.05 leftover on StackCube/PickCube
                                  # went unnoticed for a while, see Current status)
actor_success_only: true         # if true, actor batch is sampled only from successful transitions
utd_ratio: 20                    # gradient updates per new transition collected — the update
                                  # that unlocked PushCube's positive result (see Current status);
                                  # higher was tested and helped further, bounded by GPU memory,
                                  # not by an inherent instability at this value

reward_mode: "sparse"            # explicit now for a reason — see Current status. Also has a
                                  # code-level fallback (getattr(cfg, "reward_mode", "sparse")
                                  # in maniskill_env.py) so omitting this field entirely still
                                  # can't silently reintroduce the old bug.

offline_ratio: 0.5               # fraction of EACH TRAINING BATCH drawn from the offline demo
                                  # buffer during sampling. Means exactly this at every value,
                                  # including 0.0 — does NOT by itself control whether demos are
                                  # used at all (see rl_seed_demos_online below; this used to be
                                  # conflated, see Changelog).
rl_seed_demos_online: false      # if true, ALSO seed demos directly into the online replay
                                  # buffer (matches the original paper's own single-buffer
                                  # convention). With this false AND offline_ratio: 0.0, demos
                                  # are not used anywhere — pure off-policy training on collected
                                  # rollout samples only. PushCube's successful run used this
                                  # exact combination.

checkpoint_buffer: true          # save every collected transition to disk (buffers/) so a
                                  # preempted run's online replay buffer (and eval/success_rate's
                                  # rolling window) survive a resume instead of restarting empty
                                  # — see Changelog. Real disk cost: roughly 300KB/transition,
                                  # dominated by the two camera images; no automatic pruning, so
                                  # a full 120K-step run leaves ~35GB in buffers/ that you'll want
                                  # to clean up manually once a run is done being resumed.

# Rigorous, held-out, deterministic evaluation — see Current status.
rl_eval_interval: 20000          # 0 disables this feature entirely
rl_eval_episodes: 200            # reduce toward 50 if this meaningfully slows down training
rl_eval_seed: 42                 # master seed for the fixed episode list

# Categorical critic (XQC/XQCfD-style, bounded support — see Current status)
rl_num_atoms: 101                # number of fixed support bins
rl_v_min: -10.0                  # lower bound of the fixed support (NORMALIZED reward units) —
                                  # calibrated for the OLD dense reward; needs recalibrating for
                                  # sparse reward's much narrower true range if you revisit the
                                  # categorical architecture (see Current status)
rl_v_max: 20.0                   # upper bound of the fixed support, same caveat as above
rl_reward_scale_decay: 0.99      # EMA decay for the running reward-RMS estimate — this
                                  # normalization is what caused the self-reinforcing loop under
                                  # sparse reward, see Current status; bypass with
                                  # use_reward_normalization: false
use_reward_normalization: true   # set false for the categorical architecture under sparse
                                  # reward — see Current status

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

**No RL checkpoints are published here yet.** PushCube's RL run now beats its
SFT baseline (see Current status above) — worth publishing once the
StackCube/PickCube re-runs with the corrected `rl_edit_scale` are in and the
full picture across all three tasks is settled, rather than publishing one
result at a time.

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
