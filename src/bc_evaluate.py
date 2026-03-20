"""
BC Pretraining + PPO Fine-tuning Pipeline

Steps:
1. Load expert demos from HDF5 robomimic dataset
2. Pretrain SB3's ActorCriticPolicy with BC (MSE on actions)
3. Initialize PPO with the BC-pretrained policy weights
4. Fine-tune with PPO
5. Evaluate and save results
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import matplotlib.pyplot as plt

# import robosuite.utils.macros as macros
# macros.SIMULATION_WARNINGS = False

from robomimic.robomimic_env import (
    RobomimicEnvWrapper,
    load_robomimic_trajectories,
    evaluate_policy_robomimic
)
from visualize import visualize_trajectories_videos

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback



# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path",  type=str, default="/home/yash/Stanford/CS234/project/cs234/low_dim_v15.hdf5",           help="Path to robomimic HDF5 dataset")
parser.add_argument("--save_dir",      type=str, default="./",   help="Directory to save all outputs")
parser.add_argument("--num_demos",     type=int, default=50,             help="Number of demos to load (None = all)")

# BC args
parser.add_argument("--bc_epochs",     type=int,   default=1000,            help="BC pretraining epochs")
parser.add_argument("--bc_lr",         type=float, default=3e-4,           help="BC learning rate")
parser.add_argument("--bc_batch_size", type=int,   default=256,            help="BC batch size")

# PPO args
parser.add_argument("--ppo_timesteps", type=int,   default=500_000,        help="PPO fine-tuning timesteps")
parser.add_argument("--n_steps",       type=int,   default=2048,           help="PPO n_steps per rollout")

# Env args
parser.add_argument("--horizon",       type=int,   default=500,            help="Episode horizon")
parser.add_argument("--num_eval_eps",  type=int,   default=10,             help="Number of eval episodes")
parser.add_argument("--full_loss",     action="store_true",                help="Use full dense reward")

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

print("\n[1/4] Creating environment...")
env = RobomimicEnvWrapper(
    env_name="Lift",
    robots="Panda",
    use_camera_obs=False,
    render_mode="rgb_array",
    has_renderer=False,
    use_object_obs=True,
    horizon=args.horizon,
    full_loss=args.full_loss,
)
print(f"  Obs space:    {env.observation_space.shape}")
print(f"  Action space: {env.action_space.shape}")


# ---------------------------------------------------------------------------
# Load expert demonstrations
# ---------------------------------------------------------------------------

print("\n[2/4] Loading expert demonstrations...")
trajectories = load_robomimic_trajectories(
    dataset_path=args.dataset_path,
    num_trajectories=args.num_demos,
    verbose=True,
)

# Flatten all trajectories into (N, obs_dim) and (N, action_dim)
all_obs     = np.concatenate(trajectories["observations"], axis=0)  # (N, obs_dim)
all_actions = np.concatenate(trajectories["actions"],      axis=0)  # (N, action_dim)

print(f"  Total transitions: {len(all_obs)}")

obs_tensor    = torch.tensor(all_obs,     dtype=torch.float32)
action_tensor = torch.tensor(all_actions, dtype=torch.float32)

dataset    = TensorDataset(obs_tensor, action_tensor)
dataloader = DataLoader(dataset, batch_size=args.bc_batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# BC Pretraining
# ---------------------------------------------------------------------------

print("\n[3/4] BC Pretraining...")

# Build ActorCriticPolicy with the same spaces as the env
bc_policy = ActorCriticPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=lambda _: args.bc_lr,
)



# Create PPO model — SB3 will build its own ActorCriticPolicy internally
ppo_model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=args.n_steps,
    tensorboard_log=os.path.join(args.save_dir, "tensorboard"),
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
)

# Transplant BC-pretrained weights into PPO's policy
# Only load weights that match (policy network; value head stays random)
bc_state    = bc_policy.state_dict()
ppo_state   = ppo_model.policy.state_dict()

matched = {k: v for k, v in bc_state.items() if k in ppo_state and ppo_state[k].shape == v.shape}
ppo_state.update(matched)
ppo_model.policy.load_state_dict(ppo_state)





# # ---------------------------------------------------------------------------
# # Evaluation
# # ---------------------------------------------------------------------------

# print("\nEvaluating final policy...")

class SB3Agent:
    def __init__(self, model):
        self.model = model

    def get_action(self, obs: np.ndarray, deterministic: bool = True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

agent = SB3Agent(ppo_model)

results = evaluate_policy_robomimic(
    env, agent,
    num_episodes=args.num_eval_eps,
    deterministic=True,
    verbose=True,
    visualize=True,
)

# # Save eval videos
video_dir = os.path.join(args.save_dir, "videos")
visualize_trajectories_videos(results["episodes"], out_dir=video_dir)
print(f"  Videos saved to: {video_dir}")

# # Save metrics
# metrics_path = os.path.join(args.save_dir, "final_metrics.txt")
# with open(metrics_path, "w") as f:
#     f.write(f"Success rate:   {results['success_rate']:.2%}\n")
#     f.write(f"Average reward: {results['avg_reward']:.4f}\n")
#     f.write(f"Average steps:  {results['avg_steps']:.1f}\n")
# print(f"  Metrics saved to: {metrics_path}")

# env.close()
# print("\nDone.")