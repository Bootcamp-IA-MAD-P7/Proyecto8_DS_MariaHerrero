import pandas as pd

from src.preprocessing.pipeline import create_preprocessor


def sample_dataframe():
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female"],
            "age": [45.0, 67.0, 50.0],
            "hypertension": [0, 1, 0],
            "heart_disease": [0, 1, 0],
            "ever_married": ["Yes", "Yes", "No"],
            "work_type": ["Private", "Self-employed", "Govt_job"],
            "Residence_type": ["Urban", "Rural", "Urban"],
            "avg_glucose_level": [90.0, 150.0, 100.0],
            "bmi": [25.0, 30.0, 27.0],
            "smoking_status": [
                "never smoked",
                "formerly smoked",
                "Unknown",
            ],
        }
    )


def test_preprocessor_transforms_data():
    df = sample_dataframe()

    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == len(df)


def test_preprocessor_reuses_fitted_transformations():
    train = sample_dataframe()

    validation = pd.DataFrame(
        {
            "gender": ["Male"],
            "age": [72.0],
            "hypertension": [1],
            "heart_disease": [0],
            "ever_married": ["Yes"],
            "work_type": ["Private"],
            "Residence_type": ["Rural"],
            "avg_glucose_level": [130.0],
            "bmi": [29.0],
            "smoking_status": ["smokes"],
        }
    )

    preprocessor = create_preprocessor()

    train_transformed = preprocessor.fit_transform(train)
    validation_transformed = preprocessor.transform(validation)

    assert train_transformed.shape[1] == validation_transformed.shape[1]


def test_preprocessor_handles_missing_values():
    df = sample_dataframe()

    df.loc[0, "age"] = None
    df.loc[1, "gender"] = None

    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert not pd.isna(transformed).any()


def test_preprocessor_handles_unknown_categories():
    train = sample_dataframe()

    new_data = sample_dataframe().iloc[[0]].copy()
    new_data.loc[:, "work_type"] = "NewCategory"

    preprocessor = create_preprocessor()

    preprocessor.fit(train)
    transformed = preprocessor.transform(new_data)

    assert transformed.shape[0] == 1