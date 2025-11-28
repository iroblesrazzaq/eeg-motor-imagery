import argparse
from pathlib import Path

import yaml

from .preprocessing import preprocess_all


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess BCI IV 2a subjects to NPZ.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config.get("dataset", {})
    prep_cfg = config.get("preprocessing", {})

    preprocess_all(
        subjects=dataset_cfg.get("subjects", ["A01T"]),
        raw_dir=Path("data/raw"),
        processed_dir=Path(dataset_cfg.get("data_dir", "data/processed")),
        event_id=prep_cfg.get("event_id", {"left": 769, "right": 770, "foot": 771, "tongue": 772}),
        l_freq=prep_cfg.get("l_freq", 8.0),
        h_freq=prep_cfg.get("h_freq", 30.0),
        tmin=prep_cfg.get("tmin", 0.5),
        tmax=prep_cfg.get("tmax", 3.5),
        baseline=prep_cfg.get("baseline", None),
        normalize=prep_cfg.get("normalize", "zscore"),
    )


if __name__ == "__main__":
    main()
