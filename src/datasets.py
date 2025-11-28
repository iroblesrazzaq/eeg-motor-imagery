from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset


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
    ds = MotorImageryDataset(subjects=subjects, data_dir=data_dir)
    train_ds, val_ds, test_ds = stratified_split(ds, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
