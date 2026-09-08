from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/stroke_dataset.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    print("=== DIMENSIONES ===")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    print("\n=== COLUMNAS ===")
    print(df.columns.tolist())

    print("\n=== TIPOS DE DATOS ===")
    print(df.dtypes)

    print("\n=== PRIMERAS FILAS ===")
    print(df.head())
    print("\n=== VALORES NULOS ===")
    print(df.isnull().sum())

    print("\n=== DUPLICADOS ===")
    print(f"Filas duplicadas: {df.duplicated().sum()}")

    print("\n=== VARIABLES CATEGÓRICAS ===")

    categorical_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in categorical_columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))


    print("\n=== VARIABLES NUMÉRICAS ===")

    numerical_columns = [
        "age",
        "avg_glucose_level",
        "bmi",
    ]

    print(df[numerical_columns].describe())

    print("\n=== VARIABLES BINARIAS ===")

    binary_columns = [
        "hypertension",
        "heart_disease",
        "stroke",
    ]

    for column in binary_columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).sort_index())


    print("\n=== POSIBLES VALORES ANÓMALOS ===")

    print(f"Edad < 0: {(df['age'] < 0).sum()}")
    print(f"Edad > 120: {(df['age'] > 120).sum()}")

    print(f"Glucosa <= 0: {(df['avg_glucose_level'] <= 0).sum()}")

    print(f"BMI <= 0: {(df['bmi'] <= 0).sum()}")

    print(
        "Hypertension fuera de 0/1:",
        (~df["hypertension"].isin([0, 1])).sum(),
    )

    print(
        "Heart disease fuera de 0/1:",
        (~df["heart_disease"].isin([0, 1])).sum(),
    )

    print(
        "Stroke fuera de 0/1:",
        (~df["stroke"].isin([0, 1])).sum(),
    )


    print("\n=== DISTRIBUCIÓN DE STROKE ===")

    stroke_counts = df["stroke"].value_counts().sort_index()
    stroke_percentages = (
        df["stroke"].value_counts(normalize=True).sort_index() * 100
    )

    for value in stroke_counts.index:
        print(
            f"Stroke {value}: "
            f"{stroke_counts[value]} registros "
            f"({stroke_percentages[value]:.2f}%)"
        )

    imbalance_ratio = stroke_counts[0] / stroke_counts[1]

    print(
        f"\nRatio clase mayoritaria/minoritaria: "
        f"{imbalance_ratio:.2f}:1"
    )


if __name__ == "__main__":
    main()