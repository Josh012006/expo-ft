"""
Collect raw action statistics from pi0.5 base model on ManiSkill StackCube.
Runs N episodes and logs raw actions before any normalization.
"""
import sys, json
sys.path.insert(0, '.')

import numpy as np
from tqdm import tqdm
from expo_ft.utils.config_loader import load_task_config
from expo_ft.env.env_factory import make_env_wrapper
from expo_ft.agents.vla.pi05 import build_pi05
from expo_ft.utils.config_loader import get_sft_config_name
import openpi.training.config as _config

cfg = load_task_config('configs/task/maniskill/stack_cube.yaml')

# Build base model (no checkpoint)
model_config = _config.get_config(get_sft_config_name(cfg))
actor, actor_train_state, target_actor_params, agent_kwargs, vla_metadata = build_pi05(
    model_config, cfg, lora=True
)

env = make_env_wrapper({'env_usage': 'eval', 'video_dir': None}, cfg=cfg)

all_actions = []
n_episodes = 20

for ep in tqdm(range(n_episodes)):
    obs = env.reset()
    for step in range(50):
        import jax, jax.numpy as jnp
        # Get raw actions before unnormalization
        from expo_ft.agents.vla.pi05 import get_action_chunk
        raw_actions = get_action_chunk(actor, actor_train_state, obs, cfg, vla_metadata)
        all_actions.append(raw_actions)
        obs, _, done, _ = env.step(raw_actions[0].tolist())
        if done:
            break

env.close()

all_actions = np.concatenate(all_actions, axis=0)
print(f"Collected {len(all_actions)} action samples")
print(f"Shape: {all_actions.shape}")
print(f"Mean per dim: {all_actions.mean(axis=0).tolist()}")
print(f"Std per dim:  {all_actions.std(axis=0).tolist()}")
print(f"Min per dim:  {all_actions.min(axis=0).tolist()}")
print(f"Max per dim:  {all_actions.max(axis=0).tolist()}")
print(f"Q01 per dim:  {np.quantile(all_actions, 0.01, axis=0).tolist()}")
print(f"Q99 per dim:  {np.quantile(all_actions, 0.99, axis=0).tolist()}")

# Save
stats = {
    'mean': all_actions.mean(axis=0).tolist(),
    'std':  all_actions.std(axis=0).tolist(),
    'q01':  np.quantile(all_actions, 0.01, axis=0).tolist(),
    'q99':  np.quantile(all_actions, 0.99, axis=0).tolist(),
    'min':  all_actions.min(axis=0).tolist(),
    'max':  all_actions.max(axis=0).tolist(),
}
with open('assets/pi05_base_action_stats_maniskill.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("Saved to assets/pi05_base_action_stats_maniskill.json")
