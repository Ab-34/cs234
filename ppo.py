import numpy as np
import torch
import os
from pathlib import Path
from typing import Optional
import math
import random
import matplotlib.pyplot as plt
import argparse

import robosuite.utils as macros
macros.SIMULATION_WARNINGS = False
from robomimic.robomimic_env_copy import (
    RobomimicEnvWrapper,
    load_robomimic_trajectories,
    get_env_info_from_dataset,
    evaluate_policy_robomimic
)
from visualize import visualize_trajectories_videos, visualize_dpo_loss
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

parser = argparse.ArgumentParser(description="A simple script using argparse.")
parser.add_argument("--policy", type=str, help="MlpPolicy, CnnPolicy")
parser.add_argument("--timestep", type=int, help="Number of timesteps")
parser.add_argument("--save_path", type=str, help="Path to save files")
parser.add_argument('--full_loss', action='store_true', help='Enable verbose mode')

args = parser.parse_args()


env = RobomimicEnvWrapper(
    env_name="Lift",
    robots="Panda",
    use_camera_obs=False,
    render_mode="rgb_array",
    has_renderer=False,
    use_object_obs=True,
    full_loss=args.full_loss
)

# Box(4,) means that it is a Vector with 4 components
print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
# Discrete(2) means that there is two discrete actions
print("Action space:", env.action_space)
print("Full loss:", args.full_loss)


# The reset method is called at the beginning of an episode
obs, info = env.reset()

model = PPO(args.policy, env, verbose=1, n_steps=128, tensorboard_log="./logs/")
model.learn(total_timesteps=args.timestep, progress_bar=True)

class Agent:
    def __init__(self, a_obj):
        self.a_obj = a_obj

    def get_action(self, obs, deterministic=True):
        return self.a_obj.predict(obs,deterministic)


agent = Agent(model)
results_final = evaluate_policy_robomimic(
    env, agent, num_episodes=10, deterministic=True, verbose=True, visualize=True)


save_dir = os.path.join("./ppo_trained", f"{args.policy}_{args.timestep}_{str(args.full_loss)}")

model.save(os.path.join(save_dir,"model"))

visualize_trajectories_videos(results_final['episodes'], out_dir=os.path.join(save_dir, "videos"))

context = f"Success rate {results_final['success_rate']}, Average reward: {results_final['avg_reward']}"
with open(os.path.join(save_dir,'final_metrics.txt'), 'w') as file:
    file.write(context)
