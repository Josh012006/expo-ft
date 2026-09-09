# ExpoFT — π₀.₅ + ManiSkill RL Fine-Tuning

Sample-efficient RL fine-tuning of π₀.₅ on ManiSkill simulation tasks, using
the ExpoFT algorithm (frozen VLA + trainable residual policy + critic). Started
as a port of the original real-robot ExpoFT algorithm to simulation; along the
way turned into a deeper investigation of why RL fine-tuning was failing to
beat the SFT baseline. PushCube and StackCube now both do (see
CHANGELOG.md) — PickCube hasn't been re-tested with the fixes that got the
other two there.

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
> more stable under the corrected sparse-reward setup (see CHANGELOG.md).
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
  crashes at ~61GB on an 80GB card (see CHANGELOG.md). Don't remove this.
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
critic, current default — see CHANGELOG.md) or `EXPOLearnerCategorical`
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
  frozen VLA — see CHANGELOG.md, this needed a real fix). `job_eval.sh <venv>
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
(the goal-marker visibility patch — see CHANGELOG.md's Known issues — is required for
PickCube to be usable at all).

## Current status (September 2026)

**PushCube and StackCube both now beat their SFT baselines.** PushCube first
(see CHANGELOG.md); StackCube once its own remaining blocker was found and
fixed: `replan_steps=8`, a stale value unchanged since the start of the
project, unrelated to `edit_scale`, reward shaping, or the argmax-selection
bias already investigated. Switching StackCube to `replan_steps=4`
(PushCube's long-standing value) resolved a residual-correction collapse
pattern specific to that task. With this fix, **`edit_scale=0.2` on StackCube
reaches 0.755 success rate (deterministic, 200-episode `eval_rigorous`) after
2500 training episodes.**

A collaborator's hypothesis that a slow-reacting SAC entropy temperature was
the underlying cause was tested directly and mostly ruled out — but doing so
surfaced a real structural finding: the residual policy's exploration width
sits at a fixed point that maximizes the entropy of a Gaussian squashed
through `tanh`, universal across tasks and `edit_scale`, and insensitive to
the temperature's magnitude by construction. The actual StackCube-specific
blocker (`replan_steps`) was independent of this.

**Currently working on**: speeding up StackCube's convergence *rate* (2500
episodes to reach 0.755 is slow), via `replan_steps`/`batch_size`/`utd_ratio`.
Also added a diagnostic (`eval_rigorous/success_rate_critic_only`,
`scripts/eval_curve_critic_only.py`) to separate how much of a task's
improvement comes from the residual correction itself vs. the critic simply
picking the best of several stochastic samples from the frozen base VLA —
not yet analyzed.

See **[CHANGELOG.md](CHANGELOG.md)** for the full history, detailed findings,
and known issues.

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
(see CHANGELOG.md) — regenerate demos after changing any of these fields.

## RL hyperparameters (YAML fields)

These apply to the ExpoFT task YAMLs (`<task>_expo_ft.yaml` /
`<task>_expo_ft_categorical.yaml`).

```yaml
rl_lr: 3.0e-4                    # learning rate for critic and actor (NOTE: write scientific
                                  # notation WITH a decimal point — bare "3e-4" parses as a
                                  # string, not a float, in PyYAML; see CHANGELOG.md)
rl_discount: 0.99                # discount factor gamma for Bellman backup
rl_tau: 0.005                    # polyak averaging coefficient for critic target network
rl_init_temperature: 1.0         # initial SAC entropy temperature
rl_hidden_dims: [256, 256, 256]  # hidden layer sizes for the edit policy MLP
rl_edit_scale: 0.2               # max magnitude of residual action (paper: 0.05–0.2 by task
                                  # difficulty — double check this per-task in the YAML you're
                                  # actually using; a stale 0.05 leftover on StackCube/PickCube
                                  # went unnoticed for a while, see CHANGELOG.md)
actor_success_only: true         # if true, actor batch is sampled only from successful transitions
utd_ratio: 20                    # gradient updates per new transition collected — the update
                                  # that unlocked PushCube's positive result (see CHANGELOG.md);
                                  # higher was tested and helped further, bounded by GPU memory,
                                  # not by an inherent instability at this value

reward_mode: "sparse"            # explicit now for a reason — see CHANGELOG.md. Also has a
                                  # code-level fallback (getattr(cfg, "reward_mode", "sparse")
                                  # in maniskill_env.py) so omitting this field entirely still
                                  # can't silently reintroduce the old bug.

offline_ratio: 0.5               # fraction of EACH TRAINING BATCH drawn from the offline demo
                                  # buffer during sampling. Means exactly this at every value,
                                  # including 0.0 — does NOT by itself control whether demos are
                                  # used at all (see rl_seed_demos_online below; this used to be
                                  # conflated, see CHANGELOG.md).
rl_seed_demos_online: false      # if true, ALSO seed demos directly into the online replay
                                  # buffer (matches the original paper's own single-buffer
                                  # convention). With this false AND offline_ratio: 0.0, demos
                                  # are not used anywhere — pure off-policy training on collected
                                  # rollout samples only. PushCube's successful run used this
                                  # exact combination.

checkpoint_buffer: true          # save every collected transition to disk (buffers/) so a
                                  # preempted run's online replay buffer (and eval/success_rate's
                                  # rolling window) survive a resume instead of restarting empty
                                  # — see CHANGELOG.md. Real disk cost: roughly 300KB/transition,
                                  # dominated by the two camera images; no automatic pruning, so
                                  # a full 120K-step run leaves ~35GB in buffers/ that you'll want
                                  # to clean up manually once a run is done being resumed.

# Rigorous, held-out, deterministic evaluation — see CHANGELOG.md.
rl_eval_interval: 20000          # 0 disables this feature entirely
rl_eval_episodes: 200            # reduce toward 50 if this meaningfully slows down training
rl_eval_seed: 42                 # master seed for the fixed episode list

# Categorical critic (XQC/XQCfD-style, bounded support — see CHANGELOG.md)
rl_num_atoms: 101                # number of fixed support bins
rl_v_min: -10.0                  # lower bound of the fixed support (NORMALIZED reward units) —
                                  # calibrated for the OLD dense reward; needs recalibrating for
                                  # sparse reward's much narrower true range if you revisit the
                                  # categorical architecture (see CHANGELOG.md)
rl_v_max: 20.0                   # upper bound of the fixed support, same caveat as above
rl_reward_scale_decay: 0.99      # EMA decay for the running reward-RMS estimate — this
                                  # normalization is what caused the self-reinforcing loop under
                                  # sparse reward, see CHANGELOG.md; bypass with
                                  # use_reward_normalization: false
use_reward_normalization: true   # set false for the categorical architecture under sparse
                                  # reward — see CHANGELOG.md

# Critic pretraining (XQCfD-style warm-start on demos before RL starts)
rl_critic_pretrain_steps: 0      # 0 = disabled

# KL regularization for the edit policy (XQCfD-style, see CHANGELOG.md for
# the additive-vs-replacement caveat relative to the paper)
rl_kl_coef: 0.0                  # 0.0 = disabled (exact no-op)
rl_kl_ref_std: 1.0               # std of the fixed N(0, ref_std) reference, pre-tanh space
rl_entropy_scale: 1.0            # weight of the (separate, additive) entropy bonus —
                                  # set to 0.0 alongside rl_kl_coef for an isolated KL test
```

These are read directly by `train_pi_robo.py` and explicitly override the
corresponding fields in `configs/model/expo_ft_pi_config.py` — see
CHANGELOG.md for why this override wiring was needed (these used to be silently ignored).

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
run directory independently (see CHANGELOG.md).

These used to be CLI overrides (`--num-demos` on `run_pipeline.py`); they're
YAML-only now so a run's full configuration lives in one place.

## Published checkpoints

SFT checkpoints (LoRA, JAX/orbax format — see each model card for why no
PyTorch conversion is provided) are published on HuggingFace:

- [`josh11234/ExpoFT-Pi05-StackCube-v1-SFT`](https://huggingface.co/josh11234/ExpoFT-Pi05-StackCube-v1-SFT) (41% success on 200 held-out seeds)
- [`josh11234/ExpoFT-Pi05-PushCube-v1-SFT-62p`](https://huggingface.co/josh11234/ExpoFT-Pi05-PushCube-v1-SFT-62p) (62% success on 200 held-out seeds)
- [`josh11234/ExpoFT-Pi05-PushCube-v1-SFT-86p`](https://huggingface.co/josh11234/ExpoFT-Pi05-PushCube-v1-SFT-86p) (86% success on 200 held-out seeds)
- [`josh11234/ExpoFT-Pi05-PickCube-v1-SFT`](https://huggingface.co/josh11234/ExpoFT-Pi05-PickCube-v1-SFT) (22% success on 200 held-out seeds)

**No RL checkpoints are published here yet.** PushCube's RL run now beats its
SFT baseline (see CHANGELOG.md) — worth publishing once the
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
