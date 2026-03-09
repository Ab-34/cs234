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

from robomimic.robomimic_env_copy import (
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
bc_policy.train()

optimizer = optim.Adam(bc_policy.parameters(), lr=args.bc_lr)
mse_loss  = nn.MSELoss()

bc_losses = []

for epoch in range(args.bc_epochs):
    epoch_loss = 0.0
    num_batches = 0

    for obs_batch, action_batch in dataloader:
        optimizer.zero_grad()

        # ActorCriticPolicy.forward returns (actions, values, log_probs)
        # We use _predict / get_distribution to get the action mean for MSE
        # Using evaluate_actions gives us log_probs which we can use,
        # but for BC we want to directly regress on the mean action.
        # get_distribution gives us the distribution; we take its mean.
        dist = bc_policy.get_distribution(obs_batch)
        action_mean = dist.distribution.mean  # Gaussian mean

        loss = mse_loss(action_mean, action_batch)
        loss.backward()
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    bc_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1:>4}/{args.bc_epochs}]  BC Loss: {avg_loss:.6f}")

# Save BC loss curve
plt.figure(figsize=(8, 4))
plt.plot(bc_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("BC Pretraining Loss")
plt.tight_layout()
loss_curve_path = os.path.join(args.save_dir, "bc_loss_curve.png")
plt.savefig(loss_curve_path)
plt.close()
print(f"  BC loss curve saved to: {loss_curve_path}")

# Save BC policy weights
bc_weights_path = os.path.join(args.save_dir, "bc_policy.pth")
torch.save(bc_policy.state_dict(), bc_weights_path)
print(f"  BC weights saved to:    {bc_weights_path}")


# ---------------------------------------------------------------------------
# PPO Fine-tuning (initialized from BC weights)
# ---------------------------------------------------------------------------

print("\n[4/4] PPO Fine-tuning...")

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

print(f"  Transferred {len(matched)}/{len(ppo_state)} parameter tensors from BC → PPO")

# # Fine-tune with PPO




save_dir = os.path.join(
    "./ppo_trained/bc_ppo_1_1000_simple_reward",
    f"{args.ppo_timesteps}_"
)

os.makedirs(save_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,                 # save every 10k steps
    save_path=save_dir,
    name_prefix="ppo_model",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

# # model.learn(total_timesteps=args.timestep, progress_bar=True)
# model.learn(
#     total_timesteps=args.timestep,
#     progress_bar=True,
#     callback=checkpoint_callback
# )

ppo_model.learn(
    total_timesteps=args.ppo_timesteps,
    progress_bar=True,
    tb_log_name="ppo_bc_init",
    callback=checkpoint_callback
)

ppo_save_path = os.path.join(args.save_dir, "ppo_model")
ppo_model.save(ppo_save_path)
print(f"  PPO model saved to: {ppo_save_path}.zip")


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