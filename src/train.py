import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datasets import create_dataloaders
from .models import build_eegnet_from_config
from .utils import ensure_dir, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEGNet on motor imagery data.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        inputs = batch["eeg"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            inputs = batch["eeg"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def main():
    args = parse_args()
    config = load_config(args.config)

    train_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})

    set_seed(train_cfg.get("seed", 42))
    device = get_device(train_cfg.get("device", "auto"))
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(
        subjects=dataset_cfg.get("subjects", ["A01T"]),
        data_dir=dataset_cfg.get("data_dir", "data/processed"),
        batch_size=train_cfg.get("batch_size", 64),
        val_fraction=dataset_cfg.get("val_fraction", 0.15),
        test_fraction=dataset_cfg.get("test_fraction", 0.15),
        num_workers=train_cfg.get("num_workers", 0),
        seed=train_cfg.get("seed", 42),
        shuffle=dataset_cfg.get("shuffle", True),
    )

    # Infer input shape from first batch
    sample_batch = next(iter(train_loader))
    _, _, n_channels, n_samples = sample_batch["eeg"].shape

    model = build_eegnet_from_config(config, n_channels=n_channels, n_samples=n_samples).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    log_dir = Path(train_cfg.get("log_dir", "reports"))
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_acc = 0.0
    patience = train_cfg.get("patience", 8)
    epochs_no_improve = 0
    history = []
    start_epoch = 1

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = load_checkpoint(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            best_val_acc = ckpt.get("val_acc", 0.0)
            start_epoch = ckpt.get("epoch", 0) + 1
            epochs_no_improve = ckpt.get("epochs_no_improve", 0)
            history = ckpt.get("history", [])
            print(f"Resumed from {ckpt_path} at epoch {start_epoch}, best val acc {best_val_acc:.3f}")
        else:
            print(f"Resume path {ckpt_path} not found; starting fresh.")

    for epoch in range(start_epoch, train_cfg.get("max_epochs", 50) + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc="Val")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "config": config,
                    "epochs_no_improve": epochs_no_improve,
                    "history": history,
                },
                ckpt_dir / "best.pt",
            )
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

        if epochs_no_improve >= patience:
            print("Early stopping.")
            break

    # Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Test")
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Acc/test", test_acc, epoch)
    writer.close()

    # Save training history
    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
