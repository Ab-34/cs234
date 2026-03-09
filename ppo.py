import numpy as np
import torch
import os
from pathlib import Path
from typing import Optional
import math
import random
import matplotlib.pyplot as plt
import argparse
from stable_baselines3.common.callbacks import CheckpointCallback


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
parser.add_argument("--policy", type=str, help="MlpPolicy, CnnPolicy", default="MlpPolicy")
parser.add_argument("--timestep", type=int, help="Number of timesteps", default= 1000000)
parser.add_argument("--save_path", type=str, help="Path to save files", default = "./")
parser.add_argument('--full_loss', action='store_true', help='Enable verbose mode')
parser.add_argument("--model_path", type=str, default="/home/yash/Stanford/CS234/project/cs234/ppo_trained/rewards_new_lift_with_gripper_6_height_001_small_model_plus_lifting_trained_on_gripper_1_backup/MlpPolicy_1000000_False/ppo_model_105000_steps.zip")


args = parser.parse_args()


env = RobomimicEnvWrapper(
    env_name="Lift", # "square", "transport"
    robots="Panda",
    use_camera_obs=False,
    render_mode="rgb_array",
    has_renderer=False,
    use_object_obs=True,
    full_loss=args.full_loss
)

# env = make_vec_env(
#     lambda: RobomimicEnvWrapper(
#         env_name="Lift",
#         robots="Panda",
#         use_camera_obs=False,
#         render_mode=None,
#         has_renderer=False,
#         use_object_obs=True,
#         full_loss=args.full_loss
#     ),
#     n_envs=8
# )


# The reset method is called at the beginning of an episode
obs, info = env.reset()

# Box(4,) means that it is a Vector with 4 components
print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
# Discrete(2) means that there is two discrete actions
print("Action space:", env.action_space)
print("Full loss:", args.full_loss)


device = "cuda" if torch.cuda.is_available() else "cpu"

policy_kwargs = dict(
    net_arch=dict(
        pi=[1024, 1024, 512, 256],
        vf=[1024, 1024, 512, 256]
    ),
    activation_fn=torch.nn.ReLU,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    n_steps=2048,
    batch_size=512,
    tensorboard_log="./logs/",
)

# model = PPO(args.policy, env, verbose=1, n_steps=256, tensorboard_log="./logs/")  #small model
# model = PPO.load(args.model_path, env=env, device=device)   #pre saved model
print("Loaded pretrained model. Continuing training...")

save_dir = os.path.join(
    "./ppo_trained/rewards_new_lift_with_gripper_7_height_001_big_model_latest_error_handling_claude",
    f"{args.policy}_{args.timestep}_{str(args.full_loss)}"
)

os.makedirs(save_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,                 # save every 10k steps
    save_path=save_dir,
    name_prefix="ppo_model",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

# model.learn(total_timesteps=args.timestep, progress_bar=True)
model.learn(
    total_timesteps=args.timestep,
    progress_bar=True,
    callback=checkpoint_callback
)

class Agent:
    def __init__(self, a_obj):
        self.a_obj = a_obj

    def get_action(self, obs, deterministic=True):
        return self.a_obj.predict(obs,deterministic)


agent = Agent(model)
results_final = evaluate_policy_robomimic(
    env, agent, num_episodes=10, deterministic=True, verbose=True, visualize=True)


# save_dir = os.path.join("./ppo_trained/rewards_new_lift_with_gripper_3_height_001", f"{args.policy}_{args.timestep}_{str(args.full_loss)}")

model.save(os.path.join(save_dir,"model"))

visualize_trajectories_videos(results_final['episodes'], out_dir=os.path.join(save_dir, "videos"))

context = f"Success rate {results_final['success_rate']}, Average reward: {results_final['avg_reward']}"
with open(os.path.join(save_dir,'final_metrics.txt'), 'w') as file:
    file.write(context)
