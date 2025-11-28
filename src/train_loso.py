"""
Leave-One-Subject-Out (LOSO) Cross-Validation Training Script for Cross-Subject BCI.

This script trains an EEGNet model using LOSO cross-validation, which is the gold
standard for evaluating cross-subject generalization in BCI research.

Algorithm:
    For each of 9 subjects (A01T-A09T):
        1. Hold out subject S_i as test set
        2. Train model on remaining 8 subjects (with val split)
        3. Evaluate on held-out subject S_i
        4. Record accuracy for fold i
    
    Final metric: Mean ± Std accuracy across all 9 folds

Usage:
    python -m src.train_loso --config configs/default.yaml
    python -m src.train_loso --config configs/default.yaml --folds A01T A03T  # Specific folds
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datasets import ALL_SUBJECTS, create_loso_dataloaders, get_loso_folds
from .models import build_eegnet_from_config
from .utils import ensure_dir, get_device, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EEGNet with Leave-One-Subject-Out Cross-Validation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=str,
        default=None,
        help="Specific subjects to hold out (default: all 9 subjects).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/loso",
        help="Directory to save LOSO results.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
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


def evaluate(model, loader, criterion, device):
    """Evaluate model on given loader, return loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["eeg"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / total, correct / total


def train_fold(
    fold_subject: str,
    config: Dict,
    device: torch.device,
    output_dir: Path,
    verbose: bool = True,
) -> Dict:
    """
    Train a single LOSO fold.
    
    Args:
        fold_subject: Subject to hold out for testing
        config: Configuration dictionary
        device: Torch device
        output_dir: Directory to save fold results
        verbose: Print progress
        
    Returns:
        Dictionary with fold results (accuracies, losses, etc.)
    """
    train_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    
    # Create LOSO dataloaders
    train_loader, val_loader, test_loader = create_loso_dataloaders(
        leave_out_subject=fold_subject,
        data_dir=dataset_cfg.get("data_dir", "data/processed"),
        batch_size=train_cfg.get("batch_size", 64),
        val_fraction=dataset_cfg.get("val_fraction", 0.1),
        num_workers=train_cfg.get("num_workers", 0),
        seed=train_cfg.get("seed", 42),
        shuffle=True,
    )
    
    # Infer input shape from first batch
    sample_batch = next(iter(train_loader))
    _, _, n_channels, n_samples = sample_batch["eeg"].shape
    
    # Build model
    model = build_eegnet_from_config(config, n_channels=n_channels, n_samples=n_samples)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    
    # Setup logging
    fold_dir = output_dir / f"fold_{fold_subject}"
    ensure_dir(fold_dir)
    writer = SummaryWriter(log_dir=str(fold_dir / "tensorboard"))
    
    best_val_acc = 0.0
    patience = train_cfg.get("patience", 15)
    epochs_no_improve = 0
    history = []
    best_model_state = None
    
    max_epochs = train_cfg.get("max_epochs", 100)
    
    if verbose:
        epoch_iter = tqdm(range(1, max_epochs + 1), desc=f"Fold {fold_subject}")
    else:
        epoch_iter = range(1, max_epochs + 1)
    
    for epoch in epoch_iter:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = train_loss, train_acc
        
        # Logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
        
        if verbose:
            epoch_iter.set_postfix({
                "train_acc": f"{train_acc:.3f}",
                "val_acc": f"{val_acc:.3f}",
                "best": f"{best_val_acc:.3f}"
            })
        
        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate on test set
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    writer.add_scalar("Acc/test", test_acc, epoch)
    writer.close()
    
    # Save best model for this fold
    save_checkpoint(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "test_acc": test_acc,
            "val_acc": best_val_acc,
            "fold_subject": fold_subject,
            "config": config,
            "history": history,
        },
        fold_dir / "best_model.pt",
    )
    
    # Save fold history
    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    fold_result = {
        "fold_subject": fold_subject,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "best_val_acc": best_val_acc,
        "final_epoch": epoch,
        "n_train_trials": len(train_loader.dataset),
        "n_test_trials": len(test_loader.dataset),
    }
    
    if verbose:
        print(f"  Fold {fold_subject}: Test Acc = {test_acc:.4f} (Val Acc = {best_val_acc:.4f})")
    
    return fold_result


def main():
    args = parse_args()
    config = load_config(args.config)
    
    train_cfg = config.get("training", {})
    set_seed(train_cfg.get("seed", 42))
    device = get_device(train_cfg.get("device", "auto"))
    
    print("=" * 60)
    print("LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    
    # Determine which folds to run
    if args.folds:
        fold_subjects = args.folds
    else:
        fold_subjects = get_loso_folds()
    
    print(f"Folds: {fold_subjects}")
    print("=" * 60)
    
    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    ensure_dir(output_dir)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Run all folds
    fold_results = []
    for fold_subject in fold_subjects:
        print(f"\n{'='*40}")
        print(f"FOLD: Hold out {fold_subject}")
        print(f"Train on: {[s for s in ALL_SUBJECTS if s != fold_subject]}")
        print(f"{'='*40}")
        
        result = train_fold(
            fold_subject=fold_subject,
            config=config,
            device=device,
            output_dir=output_dir,
            verbose=True,
        )
        fold_results.append(result)
    
    # Aggregate results
    test_accs = [r["test_acc"] for r in fold_results]
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    
    summary = {
        "n_folds": len(fold_results),
        "fold_subjects": fold_subjects,
        "mean_test_acc": float(mean_acc),
        "std_test_acc": float(std_acc),
        "min_test_acc": float(np.min(test_accs)),
        "max_test_acc": float(np.max(test_accs)),
        "fold_results": fold_results,
        "config_path": args.config,
        "timestamp": timestamp,
    }
    
    # Save summary
    with open(output_dir / "loso_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print final results
    print("\n" + "=" * 60)
    print("LOSO CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"\n{'Subject':<10} {'Test Acc':>10}")
    print("-" * 20)
    for result in fold_results:
        print(f"{result['fold_subject']:<10} {result['test_acc']:>10.4f}")
    print("-" * 20)
    print(f"\n{'MEAN':<10} {mean_acc:>10.4f}")
    print(f"{'STD':<10} {std_acc:>10.4f}")
    print(f"\n>>> Cross-Subject Accuracy: {mean_acc:.2%} ± {std_acc:.2%} <<<")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

