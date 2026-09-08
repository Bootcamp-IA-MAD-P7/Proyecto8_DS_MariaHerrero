import pandas as pd
import pytest

from src.data.validation import validate_dataset


def valid_dataframe():
    return pd.DataFrame(
        {
            "gender": ["Female", "Male"],
            "age": [45.0, 67.0],
            "hypertension": [0, 1],
            "heart_disease": [0, 1],
            "ever_married": ["Yes", "Yes"],
            "work_type": ["Private", "Self-employed"],
            "Residence_type": ["Urban", "Rural"],
            "avg_glucose_level": [90.0, 150.0],
            "bmi": [25.0, 30.0],
            "smoking_status": ["never smoked", "formerly smoked"],
            "stroke": [0, 1],
        }
    )


def test_valid_dataset_passes():
    df = valid_dataframe()

    validate_dataset(df)


def test_missing_column_raises_error():
    df = valid_dataframe().drop(columns=["age"])

    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        validate_dataset(df)


def test_null_value_raises_error():
    df = valid_dataframe()
    df.loc[0, "bmi"] = None

    with pytest.raises(ValueError, match="valores nulos"):
        validate_dataset(df)


def test_duplicate_rows_raise_error():
    df = valid_dataframe()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="filas duplicadas"):
        validate_dataset(df)


def test_invalid_category_raises_error():
    df = valid_dataframe()
    df.loc[0, "gender"] = "Invalid"

    with pytest.raises(ValueError, match="Valores no permitidos"):
        validate_dataset(df)


def test_invalid_binary_value_raises_error():
    df = valid_dataframe()
    df.loc[0, "hypertension"] = 2

    with pytest.raises(ValueError, match="valores distintos de 0/1"):
        validate_dataset(df)


def test_invalid_numeric_range_raises_error():
    df = valid_dataframe()
    df.loc[0, "age"] = -5

    with pytest.raises(ValueError, match="valores menores"):
        validate_dataset(df)


def test_age_above_supported_range_raises_error():
    df = valid_dataframe()
    df.loc[0, "age"] = 121

    with pytest.raises(ValueError, match="valores mayores que 120"):
        validate_dataset(df)


@pytest.mark.parametrize(
    "field",
    ["bmi", "avg_glucose_level"],
)
def test_negative_continuous_value_raises_error(field):
    df = valid_dataframe()
    df.loc[0, field] = -0.1

    with pytest.raises(ValueError, match="valores menores que 0"):
        validate_dataset(df)


def test_zero_bmi_and_glucose_match_current_dataset_contract():
    df = valid_dataframe()
    df.loc[0, "bmi"] = 0
    df.loc[0, "avg_glucose_level"] = 0

    validate_dataset(df)


def test_target_requires_both_classes():
    df = valid_dataframe()
    df["stroke"] = 0

    with pytest.raises(ValueError, match="debe contener las clases 0 y 1"):
        validate_dataset(df)


def test_invalid_type_raises_error():
    df = valid_dataframe()
    df["age"] = ["cuarenta", "sesenta"]

    with pytest.raises(ValueError, match="Tipo incorrecto"):
        validate_dataset(df)
