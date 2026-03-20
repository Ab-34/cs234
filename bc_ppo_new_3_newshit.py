"""
BC Pretraining + PPO Fine-tuning Pipeline

Steps:
1. Load expert demos from HDF5 robomimic dataset
2. Pretrain SB3's ActorCriticPolicy with BC (MSE on actions)
3. Pre-warm the value head (actor frozen) to stabilize early advantage estimates
4. Fine-tune with PPO + BC regularization (annealed) to prevent catastrophic forgetting
5. Evaluate and save results

Key fixes over naive BC-init → PPO:
  - Conservative PPO hyperparams (lr=1e-4, clip_range=0.1, n_epochs=4)
  - Value head warm-up phase (freeze actor, train critic only for 50k steps)
  - BC regularization callback with linear annealing (lambda: 0.5 → 0)
  - Small entropy bonus (ent_coef=0.01) to encourage exploration around BC solution
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import matplotlib.pyplot as plt

from robomimic.robomimic_env_copy_abhi import (
    load_robomimic_trajectories,
    evaluate_policy_robomimic,
)

from robomimic.robomimic_env_copy_fully_new import RobomimicEnvWrapper

from visualize import visualize_trajectories_videos

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path",  type=str,   default="/home/yash/Stanford/CS234/project/cs234/low_dim_v15.hdf5")
parser.add_argument("--save_dir",      type=str,   default="./")
parser.add_argument("--num_demos",     type=int,   default=50)

# BC args
parser.add_argument("--bc_epochs",     type=int,   default=1000)
parser.add_argument("--bc_lr",         type=float, default=3e-4)
parser.add_argument("--bc_batch_size", type=int,   default=256)

# Value warmup args
parser.add_argument("--warmup_steps",  type=int,   default=50_000,   help="Steps to warm up value head (actor frozen)")

# PPO args
parser.add_argument("--ppo_timesteps", type=int,   default=500_000)
parser.add_argument("--n_steps",       type=int,   default=500)
parser.add_argument("--ppo_lr",        type=float, default=2e-5,     help="Lower LR preserves BC init")
parser.add_argument("--clip_range",    type=float, default=0.1,      help="Tighter clip preserves BC init")
parser.add_argument("--n_epochs",      type=int,   default=4,        help="Fewer passes per rollout = less drift")
parser.add_argument("--ent_coef",      type=float, default=0.01,     help="Entropy bonus to explore around BC")

# BC regularization args
parser.add_argument("--lambda_bc",     type=float, default=0.5,      help="Initial BC regularization weight")
parser.add_argument("--anneal_bc",     action="store_true", default=True, help="Linearly anneal lambda_bc to 0")

# Env args
parser.add_argument("--horizon",       type=int,   default=500)
parser.add_argument("--num_eval_eps",  type=int,   default=10)
parser.add_argument("--full_loss",     action="store_true")

args = parser.parse_args()


save_dir = os.path.join(
    args.save_dir,
    "ppo_trained/bc_ppo_1_1000_simple_reward_tushar_bcppo",
    f"{args.ppo_timesteps}_steps",
)

os.makedirs(save_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# BC Regularization Callback
# ---------------------------------------------------------------------------

class BCRegularizationCallback(BaseCallback):
    """
    After each rollout, adds a BC regularization gradient step that penalizes
    the PPO policy for drifting away from the BC-pretrained action means.

    Loss: lambda_bc * MSE(ppo_action_mean, bc_action_mean)

    lambda_bc is linearly annealed from lambda_bc_init → 0 over total_timesteps,
    so BC dominates early (stability) and RL dominates late (improvement beyond BC).
    """

    def __init__(
        self,
        bc_policy: ActorCriticPolicy,
        lambda_bc: float = 0.5,
        anneal_to_zero: bool = True,
        total_timesteps: int = 500_000,
        bc_lr: float = 1e-5,
        log_freq: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.bc_policy       = bc_policy
        self.lambda_bc_init  = lambda_bc
        self.lambda_bc       = lambda_bc
        self.anneal          = anneal_to_zero
        self.total_timesteps = total_timesteps
        self.bc_lr           = bc_lr
        self.log_freq        = log_freq
        self._bc_losses      = []
        self._bc_optimizer   = None   # created lazily after model is set
        self._steps_done     = 0      # tracks fine-tuning steps for annealing

    def _on_training_start(self) -> None:
        # Create a separate optimizer so BC reg never touches PPO's optimizer
        # state or corrupts the importance-sampling ratio inside PPO's train().
        self._bc_optimizer = optim.Adam(
            self.model.policy.parameters(), lr=self.bc_lr
        )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        """Called AFTER PPO's gradient update and before the next rollout.

        Firing here (instead of _on_rollout_end) ensures BC reg never modifies
        the policy weights between rollout collection and PPO's own train() call,
        which would corrupt the importance-sampling ratio log π_new / log π_old.
        """
        if self._bc_optimizer is None:
            return  # safety guard before training_start
        
       
        if not self.model.rollout_buffer.full:
            return


        self._steps_done += 1

        # --- Anneal lambda linearly to 0 over fine-tuning steps ---
        if self.anneal:
            progress       = self._steps_done * self.model.n_steps / self.total_timesteps
            self.lambda_bc = self.lambda_bc_init * max(0.0, 1.0 - progress)

        if self.lambda_bc < 1e-6:
            return  # Fully annealed; skip

        # --- Sample a batch from the rollout buffer ---
        rollout_batches = list(self.model.rollout_buffer.get(batch_size=256))
        if not rollout_batches:
            return

        obs_batch = rollout_batches[0].observations  # (B, obs_dim) on correct device

        # --- Compute BC regularization loss ---
        with torch.no_grad():
            bc_dist = self.bc_policy.get_distribution(obs_batch)
            bc_mean = bc_dist.distribution.mean.detach()

        ppo_dist = self.model.policy.get_distribution(obs_batch)
        ppo_mean = ppo_dist.distribution.mean

        bc_reg_loss = self.lambda_bc * F.mse_loss(ppo_mean, bc_mean)

        # --- Gradient step through the dedicated BC optimizer (NOT PPO's) ---
        self._bc_optimizer.zero_grad()
        bc_reg_loss.backward()
        self._bc_optimizer.step()

        self._bc_losses.append(bc_reg_loss.item())

        if self.verbose and self.num_timesteps % self.log_freq == 0:
            print(
                f"  [BCReg] step={self.num_timesteps:>7}  "
                f"λ={self.lambda_bc:.4f}  "
                f"loss={bc_reg_loss.item():.6f}"
            )

    def get_bc_losses(self):
        return self._bc_losses


# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------

print("\n[1/5] Creating environment...")
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

print("\n[1/5] Creating environment...")

def make_env():
    return RobomimicEnvWrapper(
        env_name="Lift",
        robots="Panda",
        use_camera_obs=False,
        render_mode="rgb_array",
        has_renderer=False,
        use_object_obs=True,
        horizon=args.horizon,
        full_loss=args.full_loss,
    )

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=False)

print(f"  Obs space:    {env.observation_space.shape}")
print(f"  Action space: {env.action_space.shape}")


# ---------------------------------------------------------------------------
# 2. Load expert demonstrations
# ---------------------------------------------------------------------------

print("\n[2/5] Loading expert demonstrations...")
trajectories = load_robomimic_trajectories(
    dataset_path=args.dataset_path,
    num_trajectories=args.num_demos,
    verbose=True,
)

all_obs     = np.concatenate(trajectories["observations"], axis=0)
all_actions = np.concatenate(trajectories["actions"],      axis=0)
print(f"  Total transitions: {len(all_obs)}")

obs_tensor    = torch.tensor(all_obs,     dtype=torch.float32)
action_tensor = torch.tensor(all_actions, dtype=torch.float32)
dataset       = TensorDataset(obs_tensor, action_tensor)
dataloader    = DataLoader(dataset, batch_size=args.bc_batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# 3. BC Pretraining
# ---------------------------------------------------------------------------

print("\n[3/5] BC Pretraining...")

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
    epoch_loss  = 0.0
    num_batches = 0

    for obs_batch, action_batch in dataloader:
        optimizer.zero_grad()

        dist        = bc_policy.get_distribution(obs_batch)
        action_mean = dist.distribution.mean

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

bc_weights_path = os.path.join(args.save_dir, "bc_policy.pth")
torch.save(bc_policy.state_dict(), bc_weights_path)
print(f"  BC weights saved to:    {bc_weights_path}")

# Move BC policy to eval mode and freeze — used only as a reference during PPO
bc_policy.eval()
for param in bc_policy.parameters():
    param.requires_grad = False


# ---------------------------------------------------------------------------
# 4. PPO Setup + BC weight transplant
# ---------------------------------------------------------------------------

print("\n[4/5] Setting up PPO...")

ppo_model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=args.n_steps,
    tensorboard_log=os.path.join(args.save_dir, "tensorboard"),
    learning_rate=args.ppo_lr,     # 1e-4  (was 3e-4)
    batch_size=16,
    n_epochs=args.n_epochs,        # 4     (was 10)
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=args.clip_range,    # 0.1   (was 0.2)
    ent_coef=args.ent_coef,        # 0.01  (was 0.0)
    vf_coef=0.5,
)

# Transplant BC weights (actor + shared extractor; value head stays random)
bc_state  = bc_policy.state_dict()
ppo_state = ppo_model.policy.state_dict()
matched   = {
    k: v for k, v in bc_state.items()
    if k in ppo_state and ppo_state[k].shape == v.shape
}
ppo_state.update(matched)
ppo_model.policy.load_state_dict(ppo_state)
print(f"  Transferred {len(matched)}/{len(ppo_state)} parameter tensors from BC → PPO")

bc_policy = bc_policy.to(ppo_model.device)
print(f"  BC policy moved to: {ppo_model.device}")

# Save a snapshot of the PPO policy right after BC weight transplant,
# before any RL updates, so we can always recover the BC-init starting point.
bc_init_save_path = os.path.join(save_dir, "ppo_bc_init")
ppo_model.save(bc_init_save_path)
print(f"  BC-init checkpoint saved to: {bc_init_save_path}.zip")


# ---------------------------------------------------------------------------
# 4a. Value head warm-up  (actor frozen, critic trains only)
# ---------------------------------------------------------------------------

print(f"\n  Warming up value head for {args.warmup_steps:,} steps (actor frozen)...")

# Freeze everything except the value head
for name, param in ppo_model.policy.named_parameters():
    print(name)
    if "value_net" not in name and "mlp_extractor.value" not in name:
        param.requires_grad = False

ppo_model.learn(
    total_timesteps=args.warmup_steps,
    progress_bar=True,
    tb_log_name="value_warmup",
    reset_num_timesteps=True,
)

# Unfreeze all parameters for full fine-tuning
for param in ppo_model.policy.parameters():
    param.requires_grad = True

# Rebuild the PPO optimizer so Adam's moment estimates start fresh for the
# actor params (they accumulated stale state while frozen during warmup).
ppo_model.policy.optimizer = optim.Adam(
    ppo_model.policy.parameters(), lr=args.ppo_lr
)
bc_init_save_path = os.path.join(save_dir, "ppo_bc_warmup")
ppo_model.save(bc_init_save_path)
print(f"  BC-init checkpoint saved to: {bc_init_save_path}.zip")
print("  Value head warmed up. All parameters unfrozen. Optimizer reset.")


# ---------------------------------------------------------------------------
# 4b. Full PPO fine-tuning with BC regularization
# ---------------------------------------------------------------------------

print(f"\n  Full PPO fine-tuning for {args.ppo_timesteps:,} steps...")


bc_reg_callback = BCRegularizationCallback(
    bc_policy=bc_policy,
    lambda_bc=args.lambda_bc,
    anneal_to_zero=args.anneal_bc,
    total_timesteps=args.ppo_timesteps,
    verbose=1,
)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=save_dir,
    name_prefix="ppo_model",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

ppo_model.learn(
    total_timesteps=args.ppo_timesteps,
    progress_bar=True,
    tb_log_name="ppo_bc_init",
    callback=[bc_reg_callback, checkpoint_callback],
    reset_num_timesteps=False,  # continue counting from warm-up
)

ppo_save_path = os.path.join(args.save_dir, "ppo_model")
ppo_model.save(ppo_save_path)
print(f"  PPO model saved to: {ppo_save_path}.zip")

# Save BC regularization loss curve
bc_reg_losses = bc_reg_callback.get_bc_losses()
if bc_reg_losses:
    plt.figure(figsize=(8, 4))
    plt.plot(bc_reg_losses)
    plt.xlabel("Rollout")
    plt.ylabel("BC Reg Loss")
    plt.title("BC Regularization Loss During PPO (annealed)")
    plt.tight_layout()
    bc_reg_curve_path = os.path.join(args.save_dir, "bc_reg_loss_curve.png")
    plt.savefig(bc_reg_curve_path)
    plt.close()
    print(f"  BC reg loss curve saved to: {bc_reg_curve_path}")


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

print("\n[5/5] Evaluating final policy...")

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

video_dir = os.path.join(args.save_dir, "videos")
visualize_trajectories_videos(results["episodes"], out_dir=video_dir)
print(f"  Videos saved to: {video_dir}")

metrics_path = os.path.join(args.save_dir, "final_metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"Success rate:   {results['success_rate']:.2%}\n")
    f.write(f"Average reward: {results['avg_reward']:.4f}\n")
    f.write(f"Average steps:  {results['avg_steps']:.1f}\n")
print(f"  Metrics saved to: {metrics_path}")

env.close()
print("\nDone.")