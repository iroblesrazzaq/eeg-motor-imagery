from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset


# Default subject list for BCI Competition IV 2a (Training files)
ALL_SUBJECTS = ["A01T", "A02T", "A03T", "A04T", "A05T", "A06T", "A07T", "A08T", "A09T"]


class MotorImageryDataset(Dataset):
    """Loads preprocessed NPZ files for specified subjects."""

    def __init__(self, subjects: Sequence[str], data_dir: str | Path, transform=None):
        self.data_dir = Path(data_dir)
        self.subjects = list(subjects)
        self.sid_to_idx = {sid: idx for idx, sid in enumerate(self.subjects)}
        self.transform = transform
        self.samples: List[Tuple[np.ndarray, int, str]] = []
        self.labels: List[int] = []
        self._load()

    def _load(self) -> None:
        for sid in self.subjects:
            path = self.data_dir / f"{sid}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing processed file: {path}")
            npz = np.load(path, allow_pickle=True)
            X = npz["X"]  # shape: trials, channels, time
            y = npz["y"]
            for trial, label in zip(X, y):
                self.samples.append((trial, int(label), sid))
                self.labels.append(int(label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trial, label, sid = self.samples[idx]
        eeg = torch.tensor(trial, dtype=torch.float32).unsqueeze(0)  # [1, C, T]
        if self.transform:
            eeg = self.transform(eeg)
        subj_idx = self.sid_to_idx[sid]
        return {"eeg": eeg, "label": torch.tensor(label), "subject": torch.tensor(subj_idx)}


def stratified_split(
    dataset: MotorImageryDataset,
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    labels = np.array(dataset.labels)
    total = len(dataset)
    if total == 0:
        raise ValueError("Dataset is empty.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(np.zeros(total), labels))

    remaining_fraction = 1.0 - test_fraction
    val_size_rel = val_fraction / remaining_fraction
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size_rel, random_state=seed)
    train_idx, val_idx = next(sss_val.split(np.zeros(len(train_val_idx)), labels[train_val_idx]))

    # Map back to original indices
    train_indices = [int(train_val_idx[i]) for i in train_idx]
    val_indices = [int(train_val_idx[i]) for i in val_idx]
    test_indices = [int(i) for i in test_idx]

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def create_dataloaders(
    subjects: Sequence[str],
    data_dir: str | Path,
    batch_size: int,
    val_fraction: float,
    test_fraction: float,
    num_workers: int = 0,
    seed: int = 42,
    shuffle: bool = True,
):
    """Create train/val/test dataloaders with stratified splits.
    
    This function splits data from the given subjects into train/val/test sets.
    For within-subject or mixed-subject training (NOT LOSO).
    """
    ds = MotorImageryDataset(subjects=subjects, data_dir=data_dir)
    train_ds, val_ds, test_ds = stratified_split(ds, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def create_loso_dataloaders(
    leave_out_subject: str,
    data_dir: str | Path,
    batch_size: int,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    shuffle: bool = True,
    all_subjects: Optional[Sequence[str]] = None,
):
    """Create dataloaders for Leave-One-Subject-Out (LOSO) cross-validation.
    
    Args:
        leave_out_subject: Subject ID to hold out for testing (e.g., "A01T")
        data_dir: Directory containing processed NPZ files
        batch_size: Batch size for dataloaders
        val_fraction: Fraction of training data to use for validation
        num_workers: Number of dataloader workers
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle training data
        all_subjects: Full list of subjects; defaults to ALL_SUBJECTS
        
    Returns:
        train_loader: DataLoader for training (all subjects except leave_out_subject)
        val_loader: DataLoader for validation (held-out portion of train subjects)
        test_loader: DataLoader for testing (only leave_out_subject)
        
    Example:
        # For fold 1: train on A02T-A09T, test on A01T
        train_loader, val_loader, test_loader = create_loso_dataloaders(
            leave_out_subject="A01T",
            data_dir="data/processed",
            batch_size=64
        )
    """
    if all_subjects is None:
        all_subjects = ALL_SUBJECTS
    
    # Validate leave_out_subject
    if leave_out_subject not in all_subjects:
        raise ValueError(
            f"leave_out_subject '{leave_out_subject}' not in subjects list: {all_subjects}"
        )
    
    # Split subjects: train subjects = all - leave_out
    train_subjects = [s for s in all_subjects if s != leave_out_subject]
    test_subjects = [leave_out_subject]
    
    # Create training dataset and split into train/val
    train_full_ds = MotorImageryDataset(subjects=train_subjects, data_dir=data_dir)
    
    # Create validation split from training data
    if val_fraction > 0:
        train_ds, val_ds, _ = stratified_split(
            train_full_ds,
            val_fraction=val_fraction,
            test_fraction=0.0,  # No test split here; test is the held-out subject
            seed=seed
        )
    else:
        train_ds = train_full_ds
        val_ds = None
    
    # Create test dataset (held-out subject)
    test_ds = MotorImageryDataset(subjects=test_subjects, data_dir=data_dir)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    ) if val_ds else None
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_loso_folds(all_subjects: Optional[Sequence[str]] = None) -> List[str]:
    """Get list of subject IDs for LOSO folds.
    
    Returns:
        List of subject IDs, one per fold. Each will be the held-out test subject.
    """
    if all_subjects is None:
        all_subjects = ALL_SUBJECTS
    return list(all_subjects)
