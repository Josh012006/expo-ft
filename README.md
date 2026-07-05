# ExpoFT — π₀.₅ + ManiSkill RL Fine-Tuning

Sample-efficient RL fine-tuning of π₀.₅ on ManiSkill simulation tasks, using
the ExpoFT algorithm (frozen VLA + trainable residual policy + critic, trained
via Bellman backup and a SAC-style actor update).

> This repo adapts the original real-DROID-robot ExpoFT codebase
> ([pd-perry/expo-ft](https://github.com/pd-perry/expo-ft)) to run entirely in
> ManiSkill simulation instead — no real robot, NUC, or spacemouse involved.
> `train_pi_robo.py` and `expo_ft/agents/` are largely unmodified from the
> reference (see Changelog for the exceptions); `run_pipeline.py` and
> `expo_ft/env/maniskill_env.py` are new, replacing the original client-server
> DROID environment with a local ManiSkill one.

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
- Use A100L (80GB), not a 40GB A100 — training peaks around 60GB.
  `job_demos.sh` doesn't need a GPU-heavy card (no model loaded there).

## Pipeline

Everything runs through `scripts/run_pipeline.py --config <task.yaml> --stage <stage>`:

| Stage | What it does | Job script |
|---|---|---|
| `demos` | Generate + convert demonstrations (motion planning → RGB replay → DROID/LeRobot format) | `job_demos.sh` |
| `sft` | Supervised fine-tuning warmup on demos | `job_sft.sh <venv> <config> [num_demos]` |
| `rl` | ExpoFT RL fine-tuning from an SFT checkpoint | `job_rl.sh <venv> <config> [num_demos] [sft_checkpoint]` |
| `all` | All of the above in sequence | — |

Evaluation: `scripts/eval_policy.py` (single checkpoint) or `scripts/eval_curve.py`
(sweeps every checkpoint in a directory on a fixed set of episode seeds, with
±1 SE error bars).

Tasks currently in use: **StackCube-v1**, **PushCube-v1**, **PickCube-v1**
(revived after the camera fixes below — see Known Issues).

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
  default); the camera-angle fix above may also resolve the gripper-occlusion
  concern raised earlier, revisiting now.

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

## Dataset-size ablation

`--num-demos N` on `run_pipeline.py` limits both SFT (LeRobot dataset episodes)
and RL (offline replay buffer demos) to the first N episodes of an
already-converted dataset — no reconversion, no config duplication.
Checkpoints auto-namespace (e.g. `..._sft_demos50`) so a limited-demo run never
collides with a full-dataset run.

## Changelog — key fixes made while adapting to ManiSkill (July 2026)

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
  (`--config`, `--task_config`, `--fsdp_devices`, `--num_data`, plus
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
verification, supports `--seed`), `diagnose_reward_timing.py` (confirms the
reward/action timing convention matches the original ExpoFT reference
implementation — intentional, not a porting bug).

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
