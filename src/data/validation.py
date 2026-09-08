EXPECTED_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]

EXPECTED_CATEGORIES = {
    "gender": {"Female", "Male"},
    "ever_married": {"Yes", "No"},
    "work_type": {
        "Private",
        "Self-employed",
        "children",
        "Govt_job",
    },
    "Residence_type": {"Urban", "Rural"},
    "smoking_status": {
        "never smoked",
        "formerly smoked",
        "smokes",
        "Unknown",
    },
}

BINARY_COLUMNS = {
    "hypertension",
    "heart_disease",
    "stroke",
}

NUMERIC_RANGES = {
    "age": (0, 120),
    "avg_glucose_level": (0, None),
    "bmi": (0, None),
}

TARGET_COLUMN = "stroke"

EXPECTED_TYPES = {
    "gender": "string",
    "age": "numeric",
    "hypertension": "integer",
    "heart_disease": "integer",
    "ever_married": "string",
    "work_type": "string",
    "Residence_type": "string",
    "avg_glucose_level": "numeric",
    "bmi": "numeric",
    "smoking_status": "string",
    "stroke": "integer",
}
import pandas as pd


def validate_columns(df: pd.DataFrame) -> list[str]:
    errors = []

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_columns = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_columns:
        errors.append(
            f"Faltan columnas obligatorias: {sorted(missing_columns)}"
        )

    if extra_columns:
        errors.append(
            f"Existen columnas no esperadas: {sorted(extra_columns)}"
        )

    return errors

def validate_types(df: pd.DataFrame) -> list[str]:
    errors = []

    for column, expected_type in EXPECTED_TYPES.items():
        if column not in df.columns:
            continue

        if expected_type == "string":
            valid = pd.api.types.is_string_dtype(df[column])

        elif expected_type == "numeric":
            valid = pd.api.types.is_numeric_dtype(df[column])

        elif expected_type == "integer":
            valid = pd.api.types.is_integer_dtype(df[column])

        else:
            valid = False

        if not valid:
            errors.append(
                f"Tipo incorrecto en '{column}'. "
                f"Esperado: {expected_type}. "
                f"Encontrado: {df[column].dtype}"
            )

    return errors

def validate_nulls(df: pd.DataFrame) -> list[str]:
    errors = []

    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]

    for column, count in columns_with_nulls.items():
        errors.append(
            f"La columna '{column}' contiene {count} valores nulos"
        )

    return errors


def validate_duplicates(df: pd.DataFrame) -> list[str]:
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        return [f"Se han detectado {duplicate_count} filas duplicadas"]

    return []


def validate_categories(df: pd.DataFrame) -> list[str]:
    errors = []

    for column, allowed_values in EXPECTED_CATEGORIES.items():
        if column not in df.columns:
            continue

        observed_values = set(df[column].dropna().unique())
        invalid_values = observed_values - allowed_values

        if invalid_values:
            errors.append(
                f"Valores no permitidos en '{column}': "
                f"{sorted(invalid_values)}"
            )

    return errors


def validate_binary_columns(df: pd.DataFrame) -> list[str]:
    errors = []

    for column in BINARY_COLUMNS:
        if column not in df.columns:
            continue

        invalid_values = set(df[column].dropna().unique()) - {0, 1}

        if invalid_values:
            errors.append(
                f"La columna '{column}' contiene valores distintos de 0/1: "
                f"{sorted(invalid_values)}"
            )

    return errors


def validate_numeric_ranges(df: pd.DataFrame) -> list[str]:
    errors = []

    for column, (min_value, max_value) in NUMERIC_RANGES.items():
        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        if min_value is not None:
            invalid_count = (df[column] < min_value).sum()

            if invalid_count > 0:
                errors.append(
                    f"La columna '{column}' contiene "
                    f"{invalid_count} valores menores que {min_value}"
                )

        if max_value is not None:
            invalid_count = (df[column] > max_value).sum()

            if invalid_count > 0:
                errors.append(
                    f"La columna '{column}' contiene "
                    f"{invalid_count} valores mayores que {max_value}"
                )

    return errors


def validate_target(df: pd.DataFrame) -> list[str]:
    errors = []

    if TARGET_COLUMN not in df.columns:
        errors.append(
            f"No existe la variable objetivo '{TARGET_COLUMN}'"
        )
        return errors

    unique_values = set(df[TARGET_COLUMN].dropna().unique())

    if unique_values != {0, 1}:
        errors.append(
            f"El target '{TARGET_COLUMN}' debe contener las clases 0 y 1. "
            f"Valores encontrados: {sorted(unique_values)}"
        )

    return errors


def validate_distribution(df: pd.DataFrame) -> list[str]:
    errors = []

    if TARGET_COLUMN not in df.columns:
        return errors

    class_counts = df[TARGET_COLUMN].value_counts()

    if len(class_counts) < 2:
        errors.append(
            f"El target '{TARGET_COLUMN}' contiene una sola clase"
        )

    return errors


def validate_dataset(df: pd.DataFrame) -> None:
    errors = []

    validators = [
        validate_columns,
        validate_types,
        validate_nulls,
        validate_duplicates,
        validate_categories,
        validate_binary_columns,
        validate_numeric_ranges,
        validate_target,
        validate_distribution,
    ]

    for validator in validators:
        errors.extend(validator(df))

    if errors:
        formatted_errors = "\n".join(
            f"- {error}" for error in errors
        )

        raise ValueError(
            "Dataset incompatible:\n"
            f"{formatted_errors}"
        )
    