import numpy as np
import torch
import os
from pathlib import Path
import argparse
from typing import Callable

import robosuite.utils as macros
macros.SIMULATION_WARNINGS = False

from robomimic.robomimic_env_copy import (
    RobomimicEnvWrapper,
    evaluate_policy_robomimic
)

from visualize import visualize_trajectories_videos, visualize_dpo_loss
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# -------------------- ARGPARSE --------------------
parser = argparse.ArgumentParser(description="PPO Training with Robomimic Lift Env")
parser.add_argument("--policy", type=str, help="MlpPolicy, CnnPolicy", default="MlpPolicy")
parser.add_argument("--timestep", type=int, help="Number of timesteps", default=500_000)
parser.add_argument("--save_path", type=str, help="Path to save files", default="./ppo_trained/parallel_training/")
parser.add_argument('--full_loss', action='store_true', help='Enable full loss computation')
parser.add_argument("--n_envs", type=int, help="Number of parallel environments", default=4)
parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model (optional)")

args = parser.parse_args()

# -------------------- ENVIRONMENT --------------------
def make_env_fn() -> Callable:
    def _init():
        env = RobomimicEnvWrapper(
            env_name="Lift",
            robots="Panda",
            use_camera_obs=False,
            render_mode="human",
            has_renderer=True,       # Disable rendering during training
            use_object_obs=True,
            full_loss=args.full_loss
        )
        return env
    return _init

# Vectorized environments for parallel training
vec_env = make_vec_env(make_env_fn, n_envs=args.n_envs)

# -------------------- MODEL --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if args.model_path is not None:
    print(f"Loading pretrained model from {args.model_path}")
    model = PPO.load(args.model_path, env=vec_env, device=device)
else:
    model = PPO(
        args.policy,
        vec_env,
        verbose=1,
        n_steps=256,
        tensorboard_log="./logs/",
        device=device
    )

print("Starting training...")
model.learn(total_timesteps=args.timestep, progress_bar=True)

# -------------------- AGENT WRAPPER --------------------
class Agent:
    def __init__(self, model_obj):
        self.model = model_obj

    def get_action(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

agent = Agent(model)

# -------------------- EVALUATION --------------------
# Use single env with rendering for evaluation
eval_env = RobomimicEnvWrapper(
    env_name="Lift",
    robots="Panda",
    use_camera_obs=False,
    render_mode="rgb_array",
    has_renderer=True,
    use_object_obs=True,
    full_loss=args.full_loss
)

results_final = evaluate_policy_robomimic(
    eval_env,
    agent,
    num_episodes=10,
    deterministic=True,
    verbose=True,
    visualize=True
)

# -------------------- SAVING --------------------
save_dir = os.path.join(args.save_path, f"{args.policy}_{args.timestep}_{args.full_loss}")
os.makedirs(save_dir, exist_ok=True)

model.save(os.path.join(save_dir, "model"))

visualize_trajectories_videos(
    results_final['episodes'],
    out_dir=os.path.join(save_dir, "videos")
)

metrics_text = f"Success rate: {results_final['success_rate']}\nAverage reward: {results_final['avg_reward']}"
with open(os.path.join(save_dir, 'final_metrics.txt'), 'w') as f:
    f.write(metrics_text)

print(f"Training complete. Results saved to {save_dir}")