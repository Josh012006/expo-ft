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
git clone <this-repo>
cd expo-ft
uv sync
```

This installs `openpi`/`openpi-client` (editable, from `expo_ft/agents/vla/openpi`)
and `mani-skill` + all other dependencies.

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

Tasks currently in use: **StackCube-v1**, **PushCube-v1**. PickCube-v1 was
tried and dropped (see Known Issues).

## Known issues / open items

- **Camera setup mismatch with π₀.₅-DROID's training distribution**: our
  external camera is front-angled, not a proper left/right third-person view,
  and PushCube has no wrist camera at all. `camera_eye_pos`/`camera_target_pos`
  in the task YAML control the external camera's pose (see below) — wrist
  camera addition is still open (it's mounted on a robot link, not freely
  repositionable the same way).
- **Resolution**: sim renders at 128×128 by default, upsampled to the model's
  native 224×224 — a likely source of degradation vs. the model's real
  pretraining data (which downsamples from high-res, not upsamples from low-res).
  Set `camera_width`/`camera_height: 224` in the task YAML to render natively;
  requires regenerating the RGB-converted demos afterward.
- **PickCube-v1 dropped**: its goal marker is hidden from sensor cameras by
  ManiSkill's default (`_hidden_objects`), and even after fixing that, the
  single fixed camera angle can let the gripper occlude a low-height goal.

## Camera configuration (YAML fields)

```yaml
camera_width: 128          # sensor camera resolution (both dims)
camera_height: 128
camera_eye_pos: [0.3, 0, 0.6]      # external camera position
camera_target_pos: [-0.1, 0, 0.1]  # what the external camera looks at
```

Read by `expo_ft/env/maniskill_env.py`, passed to ManiSkill via
`gym.make(..., sensor_configs=...)` — a config-only change, no code edits
needed to reposition or resize the external camera.
`scripts/capture_camera_comparison.py --config <task.yaml>` renders both the
sensor and human-render camera views for visual verification before
committing to a change.

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

**Tooling added:** `eval_curve.py` (checkpoint sweeps, fixed episode seeds, SE
error bars), `validate_demos_full_pipeline.py` (rigorous end-to-end demo
replay validation), `capture_camera_comparison.py` (visual camera
verification), `diagnose_reward_timing.py` (confirms the reward/action timing
convention matches the original ExpoFT reference implementation — intentional,
not a porting bug).

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
