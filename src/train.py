"""
Leave-One-Subject-Out (LOSO) Cross-Validation Training for Cross-Subject BCI.

This script trains a configurable model (EEGNet or ATCNet) using LOSO cross-validation,
the gold standard for evaluating cross-subject generalization in BCI research.

Algorithm:
    For each of 9 subjects (A01T-A09T):
        1. Hold out subject S_i as test set
        2. Train model on remaining 8 subjects (with val split)
        3. Evaluate on held-out subject S_i
        4. Record accuracy for fold i
    
    Final metric: Mean ± Std accuracy across all 9 folds

Usage:
    python -m src.train --config configs/atcnet.yaml
    python -m src.train --config configs/eegnet.yaml --folds A01T A03T  # Specific folds
    
Graceful Shutdown:
    Press Ctrl+C once to finish current fold and save progress.
    Press Ctrl+C twice to force exit immediately.
"""

import argparse
import atexit
import gc
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datasets import ALL_SUBJECTS, create_loso_dataloaders, get_loso_folds
from .models import build_model_from_config
from .utils import ensure_dir, get_device, save_checkpoint, set_seed


# Global shutdown flag for graceful interruption
_shutdown_requested = False
_force_shutdown = False


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested, _force_shutdown
    
    if _shutdown_requested:
        # Second Ctrl+C - force exit
        print("\n\n⚠️  Force shutdown requested. Cleaning up...")
        _force_shutdown = True
        _cleanup_resources()
        sys.exit(1)
    else:
        # First Ctrl+C - graceful shutdown
        _shutdown_requested = True
        print("\n\n🛑 Shutdown requested. Finishing current epoch and saving progress...")
        print("   (Press Ctrl+C again to force quit)")


def _cleanup_resources():
    """Clean up GPU/MPS resources to prevent system slowdown."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass  # Older PyTorch versions
    print("✓ Resources cleaned up")


def is_shutdown_requested() -> bool:
    """Check if graceful shutdown was requested."""
    return _shutdown_requested


def _save_partial_results(
    fold_results: List[Dict],
    all_fold_subjects: List[str],
    output_dir: Path,
    config_path: str,
    timestamp: str,
) -> None:
    """Save intermediate results after each fold for crash recovery."""
    if not fold_results:
        return
        
    test_accs = [r["test_acc"] for r in fold_results]
    completed = [r["fold_subject"] for r in fold_results]
    remaining = [s for s in all_fold_subjects if s not in completed]
    
    summary = {
        "n_folds_completed": len(fold_results),
        "n_folds_total": len(all_fold_subjects),
        "completed_subjects": completed,
        "remaining_subjects": remaining,
        "mean_test_acc": float(np.mean(test_accs)),
        "std_test_acc": float(np.std(test_accs)) if len(test_accs) > 1 else 0.0,
        "fold_results": fold_results,
        "config_path": config_path,
        "timestamp": timestamp,
        "status": "partial" if remaining else "complete",
    }
    
    with open(output_dir / "loso_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# Register cleanup on exit
atexit.register(_cleanup_resources)

# Register signal handler
signal.signal(signal.SIGINT, _signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model with Leave-One-Subject-Out Cross-Validation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eegnet.yaml",
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


def _prepare_inputs(batch, device, model_name: str):
    eeg = batch["eeg"].to(device)
    if model_name == "atcnet":
        return eeg.squeeze(1)  # Braindecode models expect [B, C, T]
    return eeg


def train_one_epoch(model, loader, criterion, optimizer, device, model_name: str):
    """Train for one epoch, return loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        inputs = _prepare_inputs(batch, device, model_name)
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


def evaluate(model, loader, criterion, device, model_name: str, return_predictions: bool = False):
    """Evaluate model on given loader, return loss and accuracy.
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Torch device
        model_name: Name of model architecture (for input preprocessing)
        return_predictions: If True, also return raw predictions and labels
        
    Returns:
        If return_predictions=False: (loss, accuracy)
        If return_predictions=True: (loss, accuracy, predictions, labels)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = _prepare_inputs(batch, device, model_name)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if return_predictions:
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    if return_predictions:
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
    return avg_loss, accuracy


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
    model_name = config.get("model", {}).get("name", "eegnet").lower()
    
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
    model = build_model_from_config(config, n_channels=n_channels, n_samples=n_samples)
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
    
    interrupted = False
    for epoch in epoch_iter:
        # Check for graceful shutdown request
        if is_shutdown_requested():
            if verbose:
                print(f"\n  ⏸️  Stopping fold {fold_subject} at epoch {epoch} (shutdown requested)")
            interrupted = True
            break
            
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, model_name)
        
        # Validation
        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, model_name)
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
    
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, model_name, return_predictions=True
    )
    
    writer.add_scalar("Acc/test", test_acc, epoch)
    writer.close()
    
    # Save test predictions for detailed analysis (confusion matrices, etc.)
    np.savez(
        fold_dir / "test_predictions.npz",
        y_pred=test_preds,
        y_true=test_labels,
        subject=fold_subject,
    )
    
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
        "interrupted": interrupted,
    }
    
    if verbose:
        status = " (interrupted)" if interrupted else ""
        print(f"  Fold {fold_subject}: Test Acc = {test_acc:.4f} (Val Acc = {best_val_acc:.4f}){status}")
    
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
    print(f"Model: {config.get('model', {}).get('name', 'eegnet')}")
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
    completed_folds = 0
    
    for fold_subject in fold_subjects:
        # Check if shutdown was requested between folds
        if is_shutdown_requested():
            print(f"\n🛑 Shutdown requested. Skipping remaining folds.")
            break
            
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
        completed_folds += 1
        
        # Save intermediate results after each fold (in case of crash/interrupt)
        _save_partial_results(fold_results, fold_subjects, output_dir, args.config, timestamp)
    
    # Aggregate results
    if not fold_results:
        print("\n⚠️  No folds completed. Exiting.")
        _cleanup_resources()
        return
        
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
    status_msg = "LOSO CROSS-VALIDATION RESULTS"
    if completed_folds < len(fold_subjects):
        status_msg += f" (PARTIAL: {completed_folds}/{len(fold_subjects)} folds)"
    print(status_msg)
    print("=" * 60)
    print(f"\n{'Subject':<10} {'Test Acc':>10} {'Status':>12}")
    print("-" * 34)
    for result in fold_results:
        status = "interrupted" if result.get("interrupted") else "complete"
        print(f"{result['fold_subject']:<10} {result['test_acc']:>10.4f} {status:>12}")
    
    # Show skipped folds
    completed_subjects = {r["fold_subject"] for r in fold_results}
    for subj in fold_subjects:
        if subj not in completed_subjects:
            print(f"{subj:<10} {'--':>10} {'skipped':>12}")
    
    print("-" * 34)
    print(f"\n{'MEAN':<10} {mean_acc:>10.4f}")
    print(f"{'STD':<10} {std_acc:>10.4f}")
    print(f"\n>>> Cross-Subject Accuracy: {mean_acc:.2%} ± {std_acc:.2%} <<<")
    
    if completed_folds < len(fold_subjects):
        print(f"\n⚠️  Partial results ({completed_folds}/{len(fold_subjects)} folds completed)")
        print(f"   Resume with: python -m src.train --folds {' '.join(s for s in fold_subjects if s not in completed_subjects)}")
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)
    
    # Final cleanup
    _cleanup_resources()


if __name__ == "__main__":
    main()
