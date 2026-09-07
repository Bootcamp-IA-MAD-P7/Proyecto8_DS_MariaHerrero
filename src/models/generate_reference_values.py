from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.explainability import (
    build_reference_values,
)
from src.models.final_model import MODEL_VERSION


TRAIN_PATH = Path("data/processed/train.csv")
OUTPUT_PATH = Path(
    "artifacts/final_model/"
    "reference_values_logreg_v1.json"
)
TARGET = "stroke"


def generate_reference_values_artifact(
    train_path=TRAIN_PATH,
    output_path=OUTPUT_PATH,
):
    train_path = Path(train_path)
    output_path = Path(output_path)

    train = pd.read_csv(train_path)
    features = train.drop(columns=[TARGET])
    reference_values = build_reference_values(
        features
    )
    serializable_values = {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in reference_values.items()
    }

    artifact = {
        "model_version": MODEL_VERSION,
        "source": {
            "dataset": train_path.as_posix(),
            "sha256": sha256(
                train_path.read_bytes()
            ).hexdigest(),
        },
        "aggregation": {
            "numeric": "median",
            "categorical": "mode_dropna_first",
        },
        "reference_values": serializable_values,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            artifact,
            indent=4,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return artifact


def main():
    generate_reference_values_artifact()


if __name__ == "__main__":
    main()
