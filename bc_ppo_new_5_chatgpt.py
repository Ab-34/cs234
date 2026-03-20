import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse

from robomimic.robomimic_env_copy_abhi import (
    RobomimicEnvWrapper,
    load_robomimic_trajectories,
    evaluate_policy_robomimic,
)

from visualize import visualize_trajectories_videos

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_path", type=str,
default="/home/yash/Stanford/CS234/project/cs234/low_dim_v15.hdf5")

parser.add_argument("--save_dir", type=str, default="./")
parser.add_argument("--num_demos", type=int, default=50)

# BC
parser.add_argument("--bc_epochs", type=int, default=500)
parser.add_argument("--bc_lr", type=float, default=3e-4)
parser.add_argument("--bc_batch_size", type=int, default=256)

# Warmup
parser.add_argument("--warmup_steps", type=int, default=50000)

# PPO
parser.add_argument("--ppo_timesteps", type=int, default=500000)
parser.add_argument("--n_steps", type=int, default=60)

# Env
parser.add_argument("--horizon", type=int, default=60)
parser.add_argument("--num_eval_eps", type=int, default=10)

args = parser.parse_args()


save_dir = os.path.join(
    args.save_dir,
    "ppo_trained/bc_ppo_stable_lift",
    f"{args.ppo_timesteps}_steps",
)

os.makedirs(save_dir, exist_ok=True)


# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------

print("\nCreating environment...")

def make_env():
    return RobomimicEnvWrapper(
        env_name="Lift",
        robots="Panda",
        use_camera_obs=False,
        has_renderer=False,
        horizon=args.horizon,
    )

env = DummyVecEnv([make_env])

# normalize both obs and reward (important)
env = VecNormalize(env, norm_obs=True, norm_reward=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading demonstrations...")

trajectories = load_robomimic_trajectories(
    dataset_path=args.dataset_path,
    num_trajectories=args.num_demos,
)

all_obs = np.concatenate(trajectories["observations"], axis=0)
all_actions = np.concatenate(trajectories["actions"], axis=0)

all_obs = env.normalize_obs(all_obs)

dataset = TensorDataset(
    torch.tensor(all_obs, dtype=torch.float32),
    torch.tensor(all_actions, dtype=torch.float32),
)

dataloader = DataLoader(
    dataset,
    batch_size=args.bc_batch_size,
    shuffle=True
)


# ------------------------------------------------------------
# BC PRETRAINING
# ------------------------------------------------------------

print("\nRunning BC pretraining...")

bc_policy = ActorCriticPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=lambda _: args.bc_lr,
)

optimizer = optim.Adam(bc_policy.parameters(), lr=args.bc_lr)
mse = nn.MSELoss()

for epoch in range(args.bc_epochs):

    total_loss = 0

    for obs, act in dataloader:

        optimizer.zero_grad()

        dist = bc_policy.get_distribution(obs)
        mean = dist.distribution.mean

        loss = mse(mean, act)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0:
        print("BC epoch", epoch, "loss:", total_loss)


# ------------------------------------------------------------
# PPO SETUP
# ------------------------------------------------------------

print("\nCreating PPO model...")

ppo_model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-5,
    n_steps=args.n_steps,
    batch_size=32,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
)


# ------------------------------------------------------------
# Transfer BC weights
# ------------------------------------------------------------

print("\nTransferring BC weights...")

bc_state = bc_policy.state_dict()
ppo_state = ppo_model.policy.state_dict()

matched = {
    k: v for k, v in bc_state.items()
    if k in ppo_state and ppo_state[k].shape == v.shape
}

ppo_state.update(matched)
ppo_model.policy.load_state_dict(ppo_state)

bc_policy = bc_policy.to(ppo_model.device)


# ------------------------------------------------------------
# VALUE WARMUP
# ------------------------------------------------------------

print("\nValue head warmup...")

for name, param in ppo_model.policy.named_parameters():

    if "value" not in name:
        param.requires_grad = False

ppo_model.learn(
    total_timesteps=args.warmup_steps,
    progress_bar=True,
    reset_num_timesteps=True,
)

for param in ppo_model.policy.parameters():
    param.requires_grad = True

ppo_model.policy.optimizer = optim.Adam(
    ppo_model.policy.parameters(),
    lr=1e-5
)

warmup_path = os.path.join(save_dir, "ppo_bc_warmup")
ppo_model.save(warmup_path)

print("Warmup model saved:", warmup_path)


# ------------------------------------------------------------
# PPO TRAINING
# ------------------------------------------------------------

print("\nRunning PPO training...")

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=save_dir,
    name_prefix="ppo_model"
)

ppo_model.learn(
    total_timesteps=args.ppo_timesteps,
    callback=[checkpoint_callback],
    progress_bar=True,
    reset_num_timesteps=False,
)

ppo_model.save(os.path.join(save_dir, "ppo_final"))
env.save(os.path.join(save_dir, "vecnormalize.pkl"))


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

print("\nEvaluating policy...")

class Agent:

    def __init__(self, model):
        self.model = model

    def get_action(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic)
        return action


agent = Agent(ppo_model)

results = evaluate_policy_robomimic(
    env,
    agent,
    num_episodes=args.num_eval_eps,
)

print("Success rate:", results["success_rate"])


video_dir = os.path.join(save_dir, "videos")

visualize_trajectories_videos(
    results["episodes"],
    out_dir=video_dir
)

print("Videos saved:", video_dir)

env.close()

print("\nDone.")