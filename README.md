EEG Motor Imagery Decoding with EEGNet
======================================

This project builds an end-to-end, production-style pipeline to decode motor imagery (left vs. right hand) from EEG using an EEGNet-style CNN, plus a classical CSP + LDA/SVM baseline. The outline below is a concrete checklist you can follow to stand up the codebase, run experiments, and report results.

Repository layout
-----------------
- `src/` — core code: datasets, preprocessing, models (EEGNet + baselines), train/eval scripts, utils.
- `configs/` — YAML configs to drive experiments (dataset splits, model hyperparams, training settings).
- `data/raw/` — downloaded EEG recordings (organized by subject/session).
- `data/processed/` — cached epochs/features (NumPy/NPZ) for faster training.
- `notebooks/` — exploratory analysis and result summaries.
- `README.md` — this guide.

Phase 0 — Environment and scaffolding
-------------------------------------
- Create the repo and directories: `src/`, `configs/`, `data/raw/`, `data/processed/`, `notebooks/`.
- Set up a Python env (conda/venv) and install: `mne`, `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, `scikit-learn`, `pyyaml`/`omegaconf`, optionally `tensorboard` or `wandb`.
- Pick a dataset (e.g., BCI Competition IV 2a/2b). Download to `data/raw/subjectXX/`.
- Add a minimal `pyproject.toml` or `requirements.txt` listing dependencies.

Phase 1 — Exploratory loading (notebook)
----------------------------------------
- In `notebooks/01_eda.ipynb`, load one subject with MNE, inspect channel names, sampling rate, event codes.
- Plot a few raw channels and PSDs to verify 8–30 Hz content.
- Identify event IDs for left/right MI and decide the epoch window relative to cue (e.g., 0.5–3.5 s).
- Prototype epoch extraction to confirm shapes: `[trials, channels, time]` and class counts.

Phase 2 — Preprocessing module
------------------------------
- Implement `src/preprocessing.py`:
  - `bandpass_filter(raw, l_freq=8.0, h_freq=30.0)`
  - `create_epochs(raw, events, event_id, tmin, tmax)`
  - `normalize_epochs(epochs_array, method="zscore")`
- Write a script/notebook (`src/preprocess_all.py` or `notebooks/02_preprocess.ipynb`) to:
  - Iterate over subjects in `data/raw/`.
  - Apply filter → epoching → label extraction → normalization.
  - Save `data/processed/subjectXX.npz` with `X`, `y`, and metadata.
- Sanity checks: class balance, absence of NaNs, consistent shapes per subject.

Phase 3 — Dataset and DataLoaders
---------------------------------
- Implement `src/datasets.py` with `MotorImageryDataset` that loads NPZ files, returns `{"eeg": tensor [1, C, T], "label": int, "subject": int}`.
- Add `create_dataloaders(train_subjects, val_subjects, test_subjects, batch_size, num_workers)` helper (can live in `datasets.py` or `train.py`).
- Smoke test: grab one batch and confirm shapes `[B, 1, C, T]`.

Phase 4 — EEGNet model
----------------------
- Implement `src/models.py` with `EEGNet`:
  - Temporal conv (Conv2d) along time.
  - Depthwise spatial conv across channels.
  - Separable conv, pooling, batch norm, dropout.
  - Final dense layer → logits for 2 classes.
- Add a unit test script/notebook to run a forward pass on random input and verify output shape `[B, 2]`.

Phase 5 — Training loop
-----------------------
- Implement `src/train.py`:
  - Parse config (YAML) or CLI flags.
  - Set seeds for reproducibility.
  - Build DataLoaders, initialize EEGNet, optimizer (Adam + weight decay), scheduler (optional), loss (CrossEntropy).
  - Train/val per epoch; track loss/accuracy; early stop on val accuracy; save best checkpoint.
  - Log metrics to CSV/JSON; optionally TensorBoard/W&B.
- Add `configs/default.yaml` with dataset windowing, batch size, lr, epochs, dropout, etc.

Phase 6 — Evaluation script
---------------------------
- Implement `src/eval.py`:
  - Load best checkpoint and config.
  - Run on held-out test split; report accuracy, confusion matrix, per-class recall, optional ROC-AUC.
  - Save metrics and plots to `reports/` (create as needed).

Phase 7 — Baseline (CSP + LDA/SVM)
----------------------------------
- Add `src/baselines.py` or functions in `preprocessing.py`:
  - Compute CSP filters (via `mne.decoding.CSP` or manual).
  - Extract log-variance features.
  - Train/evaluate LDA or SVM with the same splits.
- Notebook `notebooks/03_results_comparison.ipynb`: compare EEGNet vs. baseline (tables/plots).

Phase 8 — Visualization and diagnostics
---------------------------------------
- Plot training/validation curves from logs.
- Plot confusion matrices for test sets.
- Optional: visualize first-layer temporal kernels or spatial patterns.

Phase 9 — Documentation and polish
----------------------------------
- Flesh out this `README` with:
  - How to download data.
  - Commands to preprocess, train, and evaluate (e.g., `python -m src.preprocess_all`, `python -m src.train --config configs/default.yaml`).
  - Example results (update once available).
- Add `LICENSE` and `.gitignore`.
- Optionally add CI for lint/format and a small unit test for shapes.

Stretch ideas
-------------
- Subject-transfer experiment: train on N-1 subjects, test on the held-out subject.
- Light demo script: load a model, sample random test trials, print predicted vs. true labels.
- Streamlit or Gradio UI for quick interactive demo (optional).

Next actions
------------
- Pick dataset + download to `data/raw/`.
- Stand up env + dependency file.
- Implement preprocessing and NPZ export.
- Build EEGNet + training loop; run a first overfit-on-small-subset smoke test.
