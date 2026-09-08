from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.validation import validate_dataset


DATA_PATH = Path("data/raw/stroke_dataset.csv")
OUTPUT_DIR = Path("data/processed")

TARGET = "stroke"
RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


def split_dataset(df: pd.DataFrame):
    validate_dataset(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 80 % desarrollo / 20 % test
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Del 80 % restante obtenemos 80 % train y 20 % validation
    # Resultado final aproximado:
    # 64 % train / 16 % validation / 20 % test
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_dev,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    df = pd.read_csv(DATA_PATH)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = X_train.copy()
    train[TARGET] = y_train

    validation = X_val.copy()
    validation[TARGET] = y_val

    test = X_test.copy()
    test[TARGET] = y_test

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("=== SPLIT COMPLETADO ===")

    for name, dataset in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        positive_rate = dataset[TARGET].mean() * 100

        print(
            f"{name}: {len(dataset)} registros | "
            f"Stroke: {positive_rate:.2f}%"
        )


if __name__ == "__main__":
    main()