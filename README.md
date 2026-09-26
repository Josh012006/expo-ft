# ExpoFT — RL Fine-Tuning of VLAs

RL fine-tuning of vision-language-action models — currently π₀.₅ — using the
ExpoFT algorithm (frozen-ish VLA + trainable residual policy + critic).
Originally a simulation port of the real-robot ExpoFT codebase
([pd-perry/expo-ft](https://github.com/pd-perry/expo-ft)), running entirely
in **ManiSkill**. Now extending to contact-rich manipulation with **Isaac
Lab / FORGE**, as part of an ongoing investigation into adding tactile
sensing to VLA policies.

The algorithm itself (`expo_ft/agents/alg/`) is environment-agnostic; each
simulation backend is a self-contained wrapper + task config plugged in
through `env_factory.py`, so adding a new one doesn't touch the learner.

## Setup

```bash
git clone --recurse-submodules <this-repo>
cd expo-ft
uv sync
```

`openpi` (`expo_ft/agents/vla/openpi`) and `mani-skill`
(`expo_ft/third_party/ManiSkill`) are git submodules pointing at forks —
`--recurse-submodules` is required, or you'll get empty directories and
`uv sync` will fail. If you already cloned without it:
```bash
git submodule update --init --recursive
```
This installs `openpi`/`openpi-client` and `mani-skill` (both editable, from
the submodule paths above) + all other dependencies for the ManiSkill
pipeline.

**If `uv sync` fails to find a package**: check that `pyproject.toml`
actually lists it as a dependency — verify with:
```bash
python -c "import mani_skill.envs, torch, imageio, gymnasium, matplotlib; print('OK')"
```

### Isaac Lab / FORGE

Isaac Lab (`expo_ft/third_party/IsaacLab`) needs its own Isaac Sim / torch /
CUDA stack, kept separate from the main `.venv`:

```bash
uv venv --python 3.11 --seed .venv-isaaclab
source .venv-isaaclab/bin/activate
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
cd expo_ft/third_party/IsaacLab && ./isaaclab.sh --install none
```

Rendering requires a GPU with RT cores (e.g. RTX 8000, L40S, A6000) —
A100/H100 are not supported for Isaac Sim's rendering pipeline.

## References

### Original paper

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

### Components

- **π₀.₅** (VLA backbone) — Physical Intelligence, *"π₀.₅: A Vision-Language-Action Model with Open-World Generalization"*, [arXiv:2504.16054](https://arxiv.org/abs/2504.16054)
- **ManiSkill3** (simulation backend) — Tao et al., *"ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI"*, [arXiv:2410.00425](https://arxiv.org/abs/2410.00425)
- **Isaac Lab** (simulation backend) — Mittal et al., *"Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning"*, [arXiv:2511.04831](https://arxiv.org/abs/2511.04831)
- **FORGE** (contact-rich task suite) — Noseworthy et al., *"FORGE: Force-Guided Exploration for Robust Contact-Rich Manipulation under Uncertainty"*, [arXiv:2408.04587](https://arxiv.org/abs/2408.04587)

## Third-party components

Each of the following is a git submodule and retains its own license — only
code outside these directories is covered by this project's own license
(see `LICENSE`):

- `expo_ft/agents/vla/openpi` — fork of [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi), Apache-2.0 (see the submodule's own `LICENSE`).
- `expo_ft/third_party/ManiSkill` — fork of [ManiSkill](https://github.com/haosulab/ManiSkill) (see its `LICENSE` and `LICENSE-3RD-PARTY`).
- `expo_ft/third_party/IsaacLab` — fork of [NVIDIA's Isaac Lab](https://github.com/isaac-sim/IsaacLab), BSD-3-Clause with some components under Apache-2.0 (see the submodule's own `LICENSE`).
