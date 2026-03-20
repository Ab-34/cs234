import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import random

def plot_training_curves(history):
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history['loss'], linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Flow Matching Loss', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Velocity norms
    axes[0, 1].plot(history['velocity_norm'], label='Predicted', alpha=0.7, linewidth=2)
    axes[0, 1].plot(history['target_norm'], label='Target', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Velocity Norm', fontsize=12)
    axes[0, 1].set_title('Velocity Field Norms', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # Noise norm
    axes[1, 0].plot(history['noise_norm'], color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Noise Norm', fontsize=12)
    axes[1, 0].set_title('Base Distribution (Noise)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Loss (log scale)
    axes[1, 1].semilogy(history['loss'], linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss (log)', fontsize=12)
    axes[1, 1].set_title('Flow Matching Loss (Log Scale)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def visualize_trajectories_videos(trajectories, out_dir="videos"):
    os.makedirs(out_dir, exist_ok=True)
    sample=min(len(trajectories), 8)
    # print(trajectories)
    for i, traj in enumerate(random.sample(trajectories, sample) ):
        
        video = np.stack(traj['images'], axis=0)  # shape: (T, H, W, C)
        # print(len(video))
        video_path = os.path.join(out_dir, f"trajectory_{i:03d}.mp4")

        # Save as mp4 video using imageio
        imageio.mimsave(video_path, video, fps=20)
    print(f"Saved {sample} videos to: {out_dir}")

def visualize_dpo_loss(dpo_history, save_dir='./figures'):

    print("\n" + "="*60)
    print("DPO TRAINING COMPLETED!")
    print("="*60)

    # Check if we have raw log probs (new format) or just ratios (old format)
    has_raw_log_probs = 'log_pi_w' in dpo_history and len(dpo_history['log_pi_w']) > 0
    has_sanity_checks = 'log_pi_w_std' in dpo_history and len(dpo_history['log_pi_w_std']) > 0

    # Plot DPO training curves
    if has_sanity_checks:
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    elif has_raw_log_probs:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = np.array(axes).reshape(2, 2)  # Ensure consistent indexing

    # DPO Loss
    axes[0,0].plot(dpo_history['epoch'], dpo_history['dpo_loss'], 'b-', linewidth=2)
    axes[0,0].set_title('DPO Loss', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True, alpha=0.3)

    # Accuracy (how often policy prefers correct action)
    axes[0,1].plot(dpo_history['epoch'], dpo_history['accuracy'], 'g-', linewidth=2)
    axes[0,1].set_title('Preference Accuracy', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].set_ylim([0, 1])
    axes[0,1].grid(True, alpha=0.3)

    # Reward Margin (log_ratio_w - log_ratio_l)
    axes[1,0].plot(dpo_history['epoch'], dpo_history['reward_margin'], 'r-', linewidth=2)
    axes[1,0].set_title('Reward Margin', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Margin')
    axes[1,0].grid(True, alpha=0.3)

    # Log Ratios
    axes[1,1].plot(dpo_history['epoch'], dpo_history['log_ratio_w'], 'purple', linewidth=2, label='Winner')
    axes[1,1].plot(dpo_history['epoch'], dpo_history['log_ratio_l'], 'orange', linewidth=2, label='Loser')
    axes[1,1].set_title('Log Probability Ratios (π/π_ref)', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Log Ratio')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    # New plots for raw log probabilities
    if has_raw_log_probs:
        # Raw log probs for policy
        axes[0,2].plot(dpo_history['epoch'], dpo_history['log_pi_w'], 'purple', linewidth=2, label='Winner (π)')
        axes[0,2].plot(dpo_history['epoch'], dpo_history['log_pi_l'], 'orange', linewidth=2, label='Loser (π)')
        axes[0,2].set_title('Policy Log Probs (π)', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('Log Probability')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)

        # Raw log probs for reference (should be constant)
        axes[1,2].plot(dpo_history['epoch'], dpo_history['log_pi_ref_w'], 'purple', linewidth=2, linestyle='--', label='Winner (π_ref)')
        axes[1,2].plot(dpo_history['epoch'], dpo_history['log_pi_ref_l'], 'orange', linewidth=2, linestyle='--', label='Loser (π_ref)')
        axes[1,2].set_title('Reference Log Probs (π_ref)', fontsize=14, fontweight='bold')
        axes[1,2].set_xlabel('Epoch')
        axes[1,2].set_ylabel('Log Probability')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)

    # SANITY CHECK plots
    if has_sanity_checks:
        # Policy log prob std (proxy for entropy) - if this collapses, policy is becoming deterministic
        axes[2,0].plot(dpo_history['epoch'], dpo_history['log_pi_w_std'], 'purple', linewidth=2, label='Winner')
        axes[2,0].plot(dpo_history['epoch'], dpo_history['log_pi_l_std'], 'orange', linewidth=2, label='Loser')
        axes[2,0].set_title('⚠️ SANITY: Policy Log Prob Std (Entropy Proxy)', fontsize=14, fontweight='bold')
        axes[2,0].set_xlabel('Epoch')
        axes[2,0].set_ylabel('Std Dev')
        axes[2,0].legend()
        axes[2,0].grid(True, alpha=0.3)
        # Add warning zone
        axes[2,0].axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Warning: collapse')
        
        # Reference log prob std - should be relatively stable
        axes[2,1].plot(dpo_history['epoch'], dpo_history['log_pi_ref_w_std'], 'purple', linewidth=2, linestyle='--', label='Winner (ref)')
        axes[2,1].plot(dpo_history['epoch'], dpo_history['log_pi_ref_l_std'], 'orange', linewidth=2, linestyle='--', label='Loser (ref)')
        axes[2,1].set_title('⚠️ SANITY: Reference Log Prob Std', fontsize=14, fontweight='bold')
        axes[2,1].set_xlabel('Epoch')
        axes[2,1].set_ylabel('Std Dev')
        axes[2,1].legend()
        axes[2,1].grid(True, alpha=0.3)
        
        # Normalization check: compare first and last reference values
        ref_w_change = abs(dpo_history['log_pi_ref_w'][-1] - dpo_history['log_pi_ref_w'][0])
        ref_l_change = abs(dpo_history['log_pi_ref_l'][-1] - dpo_history['log_pi_ref_l'][0])
        axes[2,2].bar(['Ref Winner Δ', 'Ref Loser Δ'], [ref_w_change, ref_l_change], color=['purple', 'orange'])
        axes[2,2].set_title('⚠️ SANITY: Reference Drift (should be ~0)', fontsize=14, fontweight='bold')
        axes[2,2].set_ylabel('Absolute Change')
        axes[2,2].grid(True, alpha=0.3)
        # Add threshold line
        axes[2,2].axhline(y=0.1, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_dir)
    # plt.show()

    # Print final metrics
    print("\nFinal DPO Training Metrics:")
    print(f"  Final Loss: {dpo_history['dpo_loss'][-1]:.4f}")
    print(f"  Final Accuracy: {dpo_history['accuracy'][-1]:.3f}")
    print(f"  Final Reward Margin: {dpo_history['reward_margin'][-1]:.4f}")
    print(f"  Winner Log Ratio: {dpo_history['log_ratio_w'][-1]:.4f}")
    print(f"  Loser Log Ratio: {dpo_history['log_ratio_l'][-1]:.4f}")
    
    if has_raw_log_probs:
        print(f"\nRaw Log Probabilities:")
        print(f"  Policy - Winner:    {dpo_history['log_pi_w'][-1]:.4f}")
        print(f"  Policy - Loser:     {dpo_history['log_pi_l'][-1]:.4f}")
        print(f"  Reference - Winner: {dpo_history['log_pi_ref_w'][-1]:.4f}")
        print(f"  Reference - Loser:  {dpo_history['log_pi_ref_l'][-1]:.4f}")
    
    if has_sanity_checks:
        print(f"\n⚠️  SANITY CHECKS:")
        print(f"  Policy log prob std (W/L): {dpo_history['log_pi_w_std'][-1]:.4f} / {dpo_history['log_pi_l_std'][-1]:.4f}")
        print(f"  Reference drift (W/L): {ref_w_change:.4f} / {ref_l_change:.4f}")
        if dpo_history['log_pi_w_std'][-1] < 0.1 or dpo_history['log_pi_l_std'][-1] < 0.1:
            print("  ⚠️  WARNING: Policy std is very low - possible collapse!")
        if ref_w_change > 0.1 or ref_l_change > 0.1:
            print("  ⚠️  WARNING: Reference is drifting - check if it's properly frozen!")

    # Collect trajectories from learned policy
    print("\nCollecting learned policy trajectories...")
    print("NOTE: Policy uses state ONLY (no goal passed)\n")
