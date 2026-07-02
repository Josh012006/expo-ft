import h5py, numpy as np
import gymnasium as gym
import mani_skill.envs
import torch

env = gym.make('StackCube-v1', obs_mode='state', control_mode='pd_ee_delta_pose',
               num_envs=1, sim_backend='physx_cuda')

with h5py.File('demos/StackCube-v1/motionplanning/StackCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5', 'r') as f:
    keys = sorted(f.keys(), key=lambda x: int(x.split('_')[1]))
    successes = 0
    n_test = 5
    for ep_key in keys[:n_test]:
        ep = f[ep_key]
        actions_raw = np.array(ep['actions'])
        
        # Normaliser
        act_arm  = np.clip(actions_raw[:, :6] / 0.1, -1, 1)
        act_grip = np.clip((actions_raw[:, 6:7] - 0.015) / 0.025, -1, 1)
        actions_norm = np.concatenate([act_arm, act_grip], axis=-1)
        
        # Restaurer état initial
        env_states = ep['env_states']
        state_dict = {
            'actors': {k: torch.tensor(np.array(v)[0:1]) for k,v in env_states['actors'].items()},
            'articulations': {k: torch.tensor(np.array(v)[0:1]) for k,v in env_states['articulations'].items()},
        }
        env.reset()
        env.unwrapped.set_state_dict(state_dict)
        
        # Vérifier l'état après restore
        obs_after = env.unwrapped.get_obs()
        print(f'{ep_key}: actions clipped {(np.abs(actions_raw[:,:6]) > 0.1).mean():.1%}, len={len(actions_raw)}')
        
        # Rejouer
        success = False
        for t in range(len(actions_norm)):
            action_tensor = torch.tensor(actions_norm[t:t+1]).float()
            _, _, _, _, info = env.step(action_tensor)
            s = info.get('success', torch.tensor([False]))
            if hasattr(s, 'item'): s = s.item()
            if s:
                success = True
                break
        
        if success: successes += 1
        print(f'  success={success}')

print(f'Total: {successes}/{n_test}')
env.close()
