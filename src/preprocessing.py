from pathlib import Path
from typing import Dict, Iterable

import mne
import numpy as np


def bandpass_filter(raw: mne.io.BaseRaw, l_freq: float = 8.0, h_freq: float = 30.0) -> mne.io.BaseRaw:
    """Apply in-place band-pass filter; returns the same Raw for chaining."""
    raw.load_data()
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    return raw


def create_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: Dict[str, int],
    tmin: float = 0.5,
    tmax: float = 3.5,
    baseline=None,
) -> mne.Epochs:
    """Epoch raw data around events; event_id should map class names to codes."""
    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
    )


def normalize_epochs(X: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize per-channel across time."""
    if method == "zscore":
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True) + 1e-6
        return (X - mean) / std
    raise ValueError(f"Unsupported normalize method: {method}")


def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """
    Apply Euclidean Alignment (EA) to align EEG data distributions.
    
    This technique aligns the covariance matrices of EEG trials to the identity
    matrix, reducing inter-subject variability for cross-subject transfer learning.
    
    Reference:
        He & Wu (2019). "Transfer Learning for Brain-Computer Interfaces: 
        A Euclidean Space Data Alignment Approach"
    
    Args:
        X: EEG data of shape (N_trials, N_channels, N_samples)
        
    Returns:
        X_aligned: Aligned EEG data of same shape (N_trials, N_channels, N_samples)
    
    Algorithm:
        1. Compute covariance matrix C_i = X_i @ X_i.T for each trial
        2. Compute reference matrix R_bar = mean(C_i) across trials
        3. Compute whitening matrix W = R_bar^(-1/2) via eigendecomposition
        4. Transform: X_new = W @ X_old
    """
    n_trials, n_channels, n_samples = X.shape
    
    # Step 1: Compute covariance matrix for each trial
    # C_i = (1/N_samples) * X_i @ X_i.T  [shape: (n_channels, n_channels)]
    covariances = np.zeros((n_trials, n_channels, n_channels))
    for i in range(n_trials):
        trial = X[i]  # (n_channels, n_samples)
        covariances[i] = (trial @ trial.T) / n_samples
    
    # Step 2: Compute reference matrix R_bar (mean covariance)
    R_bar = covariances.mean(axis=0)  # (n_channels, n_channels)
    
    # Step 3: Compute whitening matrix W = R_bar^(-1/2)
    # Using eigendecomposition for numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(R_bar)
    
    # Regularize small/negative eigenvalues for stability
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    # W = V @ diag(eigenvalues^(-1/2)) @ V.T
    W = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    
    # Step 4: Apply whitening transform to each trial
    # X_new = W @ X_old
    X_aligned = np.zeros_like(X)
    for i in range(n_trials):
        X_aligned[i] = W @ X[i]
    
    return X_aligned


def export_subject_npz(
    raw_path: Path,
    out_path: Path,
    event_id: Dict[str, int],
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    tmin: float = 0.5,
    tmax: float = 3.5,
    baseline=None,
    normalize: str = "zscore",
    apply_euclidean_alignment: bool = True,
) -> None:
    """Load a single subject recording, preprocess, and save X/y arrays to NPZ.
    
    Args:
        raw_path: Path to raw GDF file
        out_path: Output path for NPZ file
        event_id: Mapping of event names to codes (e.g., {"left": 769, "right": 770})
        l_freq: Low cutoff for bandpass filter (Hz)
        h_freq: High cutoff for bandpass filter (Hz)
        tmin: Start time for epoch (seconds relative to event)
        tmax: End time for epoch (seconds relative to event)
        baseline: Baseline correction tuple or None
        normalize: Normalization method ("zscore" or None)
        apply_euclidean_alignment: Whether to apply EA for cross-subject alignment
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = mne.io.read_raw_gdf(raw_path, preload=True, verbose=False)
    events, mapping = mne.events_from_annotations(raw, verbose=False)

    # Map requested event codes (e.g., 769/770) to the internal integers returned by MNE
    target_event_id = {}
    for name, code in event_id.items():
        key = str(code)
        if key not in mapping:
            raise ValueError(f"Requested event code {code} ('{name}') not found in annotations mapping {mapping}")
        target_event_id[name] = mapping[key]

    raw = bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)
    epochs = create_epochs(raw, events, event_id=target_event_id, tmin=tmin, tmax=tmax, baseline=baseline)

    labels = epochs.events[:, -1]
    # Keep only requested classes (e.g., 769/770 mapped codes for hands)
    keep_mask = np.isin(labels, list(target_event_id.values()))
    epochs_data = epochs.get_data()[keep_mask]  # shape: (trials, channels, samples)
    labels = labels[keep_mask]

    # Map label codes to 0..N-1 based on event_id ordering
    code_to_idx = {code: idx for idx, code in enumerate(target_event_id.values())}
    y = np.array([code_to_idx[c] for c in labels], dtype=np.int64)

    X = epochs_data
    
    # Apply Euclidean Alignment BEFORE normalization (align to subject's own reference)
    # This standardizes covariance structure across subjects for cross-subject transfer
    if apply_euclidean_alignment:
        X = euclidean_alignment(X)
    
    if normalize:
        X = normalize_epochs(X, method=normalize)

    np.savez_compressed(out_path, X=X, y=y, info={"raw_path": str(raw_path)})


def preprocess_all(
    subjects: Iterable[str],
    raw_dir: Path,
    processed_dir: Path,
    event_id: Dict[str, int],
    l_freq: float,
    h_freq: float,
    tmin: float,
    tmax: float,
    baseline=None,
    normalize: str = "zscore",
    apply_euclidean_alignment: bool = True,
) -> None:
    """Process each subject file into an NPZ.
    
    Args:
        subjects: List of subject identifiers (e.g., ["A01T", "A02T", ...])
        raw_dir: Directory containing raw GDF files
        processed_dir: Output directory for processed NPZ files
        event_id: Mapping of event names to codes
        l_freq: Low cutoff for bandpass filter (Hz)
        h_freq: High cutoff for bandpass filter (Hz)
        tmin: Start time for epoch (seconds relative to event)
        tmax: End time for epoch (seconds relative to event)
        baseline: Baseline correction tuple or None
        normalize: Normalization method ("zscore" or None)
        apply_euclidean_alignment: Apply EA per subject for cross-subject transfer
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    for sid in subjects:
        raw_path = find_raw_file(raw_dir, sid)
        subj_name = raw_path.stem  # use actual filename stem for saved NPZ
        out_path = processed_dir / f"{subj_name}.npz"
        print(f"Processing {sid} -> {out_path}")
        export_subject_npz(
            raw_path=raw_path,
            out_path=out_path,
            event_id=event_id,
            l_freq=l_freq,
            h_freq=h_freq,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            normalize=normalize,
            apply_euclidean_alignment=apply_euclidean_alignment,
        )
    print(f"Preprocessing complete. Processed {len(list(subjects))} subjects.")


def find_raw_file(raw_dir: Path, subject_id) -> Path:
    """Resolve a subject file. Tries exact name, .gdf suffix, and glob patterns like A01*.gdf."""
    raw_dir = Path(raw_dir)
    sid = str(subject_id)
    base = sid[:-4] if sid.lower().endswith(".gdf") else sid

    candidates = [
        raw_dir / sid,
        raw_dir / f"{sid}.gdf",
        raw_dir / f"{base}.gdf",
        raw_dir / f"{base.upper()}.gdf",
    ]
    # Also try glob patterns to catch A01T/A01E variants
    candidates.extend(sorted(raw_dir.glob(f"{base}*.gdf")))

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No GDF file found for subject '{subject_id}' in {raw_dir}")
