"""
train_cnn.py

Finetunes an R3M (ResNet-50) backbone + regression head on frame_scores.json
to predict task-progress scores (0-100) using MSE loss.

Usage:
    pip install r3m torch torchvision scikit-learn tqdm
    python train_cnn.py
"""

import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from r3m import load_r3m

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SCORES_JSON   = "frame_scores.json"
CHECKPOINT_DIR = "/home/yash/Stanford/CS234/project/cs234/robomimic/robomimic/checkpoints"
BEST_CKPT     = os.path.join(CHECKPOINT_DIR, "r3m_reward_best.pt")
LAST_CKPT     = os.path.join(CHECKPOINT_DIR, "r3m_reward_last.pt")

SEED          = 42
VAL_SPLIT     = 0.1          # fraction of data held out for validation
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
FREEZE_EPOCHS = 10           # epochs to train only the head before unfreezing backbone
UNFREEZE_LR   = 1e-5        # lr for backbone after unfreezing

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# R3M image stats (ImageNet, same as R3M preprocessing)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class FrameScoreDataset(Dataset):
    def __init__(self, items: list[tuple[str, float]], transform):
        self.items = items       # list of (img_path, score_normalized)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, score = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(score, dtype=torch.float32)


def build_datasets(scores_json: str):
    with open(scores_json) as f:
        raw = json.load(f)

    # Normalize paths and scores
    items = []
    for path, score in raw.items():
        path = path.replace("/", os.sep)
        if not os.path.exists(path):
            print(f"  Warning: missing file {path} — skipping.")
            continue
        items.append((path, float(score) / 100.0))  # normalize to [0, 1]

    if not items:
        raise RuntimeError(f"No valid items found in {scores_json}.")

    random.shuffle(items)
    n_val = max(1, int(len(items) * VAL_SPLIT))
    train_items = items[n_val:]
    val_items   = items[:n_val]

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = FrameScoreDataset(train_items, train_tf)
    val_ds   = FrameScoreDataset(val_items,   val_tf)
    return train_ds, val_ds


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class R3MRewardModel(nn.Module):
    """R3M backbone (frozen or finetuned) + MLP regression head → score in [0,1]."""

    def __init__(self):
        super().__init__()
        r3m = load_r3m("resnet50")   # 2048-dim embedding
        r3m.eval()
        self.backbone = r3m

        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),            # output in [0, 1]; multiply by 100 at inference
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        # R3M expects uint8 images in [0,255]; undo normalization and rescale
        # Actually r3m.module handles its own normalization — we pass raw pixels.
        # Since our DataLoader already normalizes with ImageNet stats for the head,
        # we use the backbone's internal preprocessing instead. See note below.
        with torch.set_grad_enabled(self.backbone.training):
            emb = self.backbone(x)   # (B, 2048)
        return self.head(emb).squeeze(-1)   # (B,)


# ──────────────────────────────────────────────
# R3M preprocessing note
# ──────────────────────────────────────────────
# R3M's load_r3m() wraps the ResNet in an R3M module that applies its own
# normalization internally. We therefore feed it the raw [0,1] float tensor
# (no ImageNet normalization). We keep separate val/train transforms that
# do NOT apply Normalize for the backbone path, so we split the pipeline.

def build_datasets_r3m(scores_json: str):
    """Same as build_datasets but without ImageNet normalization (R3M handles it)."""
    with open(scores_json) as f:
        raw = json.load(f)

    items = []
    for path, score in raw.items():
        path = path.replace("/", os.sep)
        if not os.path.exists(path):
            print(f"  Warning: missing file {path} — skipping.")
            continue
        items.append((path, float(score) / 100.0))

    if not items:
        raise RuntimeError(f"No valid items found in {scores_json}.")

    random.seed(SEED)
    random.shuffle(items)
    n_val = max(1, int(len(items) * VAL_SPLIT))
    train_items = items[n_val:]
    val_items   = items[:n_val]

    # R3M expects 3 x H x W float tensor with pixels in [0, 255]
    # We'll pass them as float after ToTensor (which gives [0,1]) and
    # multiply by 255 in the forward pass via a small wrapper.
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),          # → [0, 1]
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = FrameScoreDataset(train_items, train_tf)
    val_ds   = FrameScoreDataset(val_items,   val_tf)
    return train_ds, val_ds


class R3MRewardModelFixed(nn.Module):
    """R3M backbone + MLP head. Passes pixels scaled to [0,255] as R3M expects."""

    def __init__(self):
        super().__init__()
        from r3m import load_r3m
        r3m = load_r3m("resnet50")
        self.backbone = r3m

        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        # x: (B, 3, 224, 224) float in [0, 1] — scale to [0, 255] for R3M
        x255 = x * 255.0
        emb = self.backbone(x255)           # (B, 2048)
        return self.head(emb).squeeze(-1)   # (B,)


# ──────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, train: bool):
    model.train(train)
    total_loss = 0.0
    total_mae  = 0.0

    with torch.set_grad_enabled(train):
        for imgs, targets in loader:
            imgs    = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(imgs)
            loss  = criterion(preds, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(imgs)
            # MAE in original 0-100 scale
            total_mae  += (preds - targets).abs().sum().item() * 100.0

    n = len(loader.dataset)
    return total_loss / n, total_mae / n


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading dataset from {SCORES_JSON} ...")
    train_ds, val_ds = build_datasets_r3m(SCORES_JSON)
    print(f"  Train: {len(train_ds)} frames | Val: {len(val_ds)} frames")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print("Building R3M model ...")
    model = R3MRewardModelFixed().to(DEVICE)
    model.freeze_backbone()

    criterion = nn.MSELoss()

    # Phase 1: train head only
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS)

    best_val_loss = float("inf")

    print(f"\n── Phase 1: training head only for {FREEZE_EPOCHS} epochs ──")
    for epoch in range(1, FREEZE_EPOCHS + 1):
        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss,   val_mae   = run_epoch(model, val_loader,   criterion, optimizer, train=False)
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{FREEZE_EPOCHS} | "
              f"train_loss={train_loss:.5f} train_MAE={train_mae:.2f} | "
              f"val_loss={val_loss:.5f} val_MAE={val_mae:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_CKPT)

    # Phase 2: unfreeze backbone, lower lr
    print(f"\n── Phase 2: finetuning full model for {NUM_EPOCHS - FREEZE_EPOCHS} epochs ──")
    model.unfreeze_backbone()
    optimizer = torch.optim.Adam(model.parameters(), lr=UNFREEZE_LR, weight_decay=WEIGHT_DECAY)
    remaining = NUM_EPOCHS - FREEZE_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

    for epoch in range(FREEZE_EPOCHS + 1, NUM_EPOCHS + 1):
        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss,   val_mae   = run_epoch(model, val_loader,   criterion, optimizer, train=False)
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.5f} train_MAE={train_mae:.2f} | "
              f"val_loss={val_loss:.5f} val_MAE={val_mae:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_CKPT)
            print(f"    ✓ Saved best checkpoint (val_loss={best_val_loss:.5f})")

    torch.save(model.state_dict(), LAST_CKPT)
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.5f}")
    print(f"  Best checkpoint : {BEST_CKPT}")
    print(f"  Last checkpoint : {LAST_CKPT}")


# ──────────────────────────────────────────────
# Inference helper (importable)
# ──────────────────────────────────────────────

def load_reward_model(ckpt_path: str = BEST_CKPT) -> R3MRewardModelFixed:
    """Load a trained reward model for inference."""
    model = R3MRewardModelFixed().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


def predict_score(model: R3MRewardModelFixed, img_path: str) -> float:
    """Return a 0-100 score for a single image."""
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score = model(x).item() * 100.0
    return round(score, 2)


if __name__ == "__main__":
    main()
