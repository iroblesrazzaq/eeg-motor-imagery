import argparse
import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

from .datasets import create_dataloaders
from .models import build_eegnet_from_config
from .utils import get_device, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EEGNet checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to checkpoint.")
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("training", {})

    device = get_device(train_cfg.get("device", "auto"))

    _, _, test_loader = create_dataloaders(
        subjects=dataset_cfg.get("subjects", ["A01T"]),
        data_dir=dataset_cfg.get("data_dir", "data/processed"),
        batch_size=train_cfg.get("batch_size", 64),
        val_fraction=dataset_cfg.get("val_fraction", 0.15),
        test_fraction=dataset_cfg.get("test_fraction", 0.15),
        num_workers=train_cfg.get("num_workers", 0),
        seed=train_cfg.get("seed", 42),
        shuffle=False,
    )

    sample_batch = next(iter(test_loader))
    _, _, n_channels, n_samples = sample_batch["eeg"].shape
    model = build_eegnet_from_config(config, n_channels=n_channels, n_samples=n_samples).to(device)

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", leave=False):
            inputs = batch["eeg"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)

    results = {"accuracy": acc, "confusion_matrix": cm.tolist(), "classification_report": report}
    out_path = Path(train_cfg.get("log_dir", "reports")) / "test_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Test accuracy: {acc:.3f}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
