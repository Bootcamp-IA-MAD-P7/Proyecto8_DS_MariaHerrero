import pandas as pd

from src.data.split_dataset import TARGET, split_dataset


def create_dataset():
    return pd.DataFrame(
        {
            "gender": ["Female", "Male"] * 50,
            "age": [float(value) for value in range(100)],
            "hypertension": [0, 1] * 50,
            "heart_disease": [0, 0, 1, 0] * 25,
            "ever_married": ["Yes", "No"] * 50,
            "work_type": [
                "Private",
                "Self-employed",
                "children",
                "Govt_job",
            ] * 25,
            "Residence_type": ["Urban", "Rural"] * 50,
            "avg_glucose_level": [
                70.0 + value for value in range(100)
            ],
            "bmi": [20.0 + value / 10 for value in range(100)],
            "smoking_status": [
                "never smoked",
                "formerly smoked",
                "smokes",
                "Unknown",
            ] * 25,
            "stroke": [0] * 95 + [1] * 5,
        }
    )


def test_split_has_expected_sizes_and_excludes_target():
    df = create_dataset()

    X_train, X_val, X_test, _, _, _ = split_dataset(df)

    assert len(X_train) == 64
    assert len(X_val) == 16
    assert len(X_test) == 20
    assert TARGET not in X_train.columns
    assert TARGET not in X_val.columns
    assert TARGET not in X_test.columns


def test_split_is_reproducible():
    df = create_dataset()

    first_split = split_dataset(df)
    second_split = split_dataset(df)

    for first, second in zip(first_split, second_split):
        assert first.equals(second)


def test_split_has_no_overlapping_rows():
    df = create_dataset()

    X_train, X_val, X_test, _, _, _ = split_dataset(df)

    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)
    assert set(X_train.index) | set(X_val.index) | set(
        X_test.index
    ) == set(df.index)


def test_split_preserves_class_distribution():
    df = create_dataset()

    _, _, _, y_train, y_val, y_test = split_dataset(df)

    original_rate = df[TARGET].mean()

    for split in [y_train, y_val, y_test]:
        assert abs(split.mean() - original_rate) < 0.02
