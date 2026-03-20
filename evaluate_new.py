import numpy as np
import torch
import os
import argparse

import robosuite.utils as macros
macros.SIMULATION_WARNINGS = False

from robomimic.robomimic_env_copy_abhi_evaluate import (
    RobomimicEnvWrapper,
    evaluate_policy_robomimic
)

import visualize_yash

from stable_baselines3 import PPO


# GOOD EVALUATE

# =========================
# ARGPARSE
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, default="MlpPolicy")
i = 12
parser.add_argument("--model_path", type=str, default="/home/yash/Stanford/CS234/project/cs234/ppo_trained/bc_ppo_4_try9_60_steps_mine_full_complex_old3_yes_grasp/500000_steps/ppo_model_50100_steps.zip")
parser.add_argument("--full_loss", action="store_true")
parser.add_argument("--episodes", type=int, default=6)
parser.add_argument("--bc_lr",         type=float, default=3e-4,           help="BC learning rate")

args = parser.parse_args()


# =========================
# ENV
# =========================
env = RobomimicEnvWrapper(
    env_name="Lift",
    robots="Panda",
    use_camera_obs=False,
    render_mode="rgb_array",
    has_renderer=True,
    use_object_obs=True,
    full_loss=args.full_loss,
    horizon = 100
)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)


# =========================
# LOAD MODEL
# =========================
model = PPO.load(args.model_path, env=env)


check_bc = False
if check_bc:

    from stable_baselines3.common.policies import ActorCriticPolicy

    # Build ActorCriticPolicy with the same spaces as the env
    bc_policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: args.bc_lr,
    )


    bc_policy.load_state_dict(torch.load('/home/yash/Stanford/CS234/project/cs234/bc_policy.pth'))

    # Transplant BC-pretrained weights into PPO's policy
    # Only load weights that match (policy network; value head stays random)
    bc_state    = bc_policy.state_dict()
    ppo_state   = model.policy.state_dict()

    matched = {k: v for k, v in bc_state.items() if k in ppo_state and ppo_state[k].shape == v.shape}
    ppo_state.update(matched)
    model.policy.load_state_dict(ppo_state)


# =========================
# AGENT WRAPPER (same as yours)
# =========================
class Agent:
    def __init__(self, model):
        self.model = model

    def get_action(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action


agent = Agent(model)


# =========================
# EVALUATE
# =========================
results = evaluate_policy_robomimic(
    env,
    agent,
    num_episodes=args.episodes,
    deterministic=True,
    verbose=True,
    visualize=True   # <-- This shows rendering
)



visualize_yash.visualize_trajectories_videos(results['episodes'], out_dir=f"/home/yash/Stanford/CS234/project/cs234/visualized_videos/bc_ppo_dense_ckpt_{i}")

print("\n========== FINAL RESULTS ==========")
print("Success rate:", results["success_rate"])
print("Average reward:", results["avg_reward"])