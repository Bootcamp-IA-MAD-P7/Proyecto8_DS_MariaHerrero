from pydantic import ValidationError

from src.clinical_safety import (
    CLINICAL_DISCLAIMER,
    SCORE_LABEL,
    safe_classification_label,
)
from src.api.schemas import (
    PredictionRequest,
)
from src.api.services.model_service import (
    ModelService,
)
from src.api.services.prediction_service import (
    PredictionService,
)


def ask_choice(
    label,
    options,
):
    while True:
        print(f"\n{label}")

        for index, option in enumerate(
            options,
            start=1,
        ):
            print(
                f"{index}. {option}"
            )

        value = input(
            "Selecciona una opción: "
        ).strip()

        try:
            index = int(value) - 1

            if 0 <= index < len(
                options
            ):
                return options[index]

        except ValueError:
            pass

        print(
            "Entrada no válida. "
            "Inténtalo de nuevo."
        )


def ask_float(
    label,
):
    while True:
        value = input(
            f"{label}: "
        ).strip()

        try:
            return float(value)

        except ValueError:
            print(
                "Debes introducir "
                "un número válido."
            )


def ask_binary(
    label,
):
    while True:
        value = input(
            f"{label} (0 = No, 1 = Sí): "
        ).strip()

        if value in {
            "0",
            "1",
        }:
            return int(value)

        print(
            "Introduce únicamente "
            "0 o 1."
        )


def collect_patient_data():
    print(
        "\n=== Stroke Risk AI ==="
    )

    print(
        "\nIntroduce los datos "
        "de la evaluación."
    )

    origin = ask_choice(
        "Origen de los datos:",
        [
            "professional",
            "self_reported",
        ],
    )

    gender = ask_choice(
        "Sexo:",
        [
            "Female",
            "Male",
            "Other",
        ],
    )

    age = ask_float(
        "Edad"
    )

    hypertension = ask_binary(
        "Hipertensión"
    )

    heart_disease = ask_binary(
        "Enfermedad cardíaca"
    )

    ever_married = ask_choice(
        "¿Ha estado casado/a?",
        [
            "Yes",
            "No",
        ],
    )

    work_type = ask_choice(
        "Tipo de trabajo:",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked",
        ],
    )

    residence_type = ask_choice(
        "Tipo de residencia:",
        [
            "Urban",
            "Rural",
        ],
    )

    avg_glucose_level = ask_float(
        "Nivel medio de glucosa"
    )

    bmi = ask_float(
        "BMI"
    )

    smoking_status = ask_choice(
        "Estado respecto al tabaco:",
        [
            "formerly smoked",
            "never smoked",
            "smokes",
            "Unknown",
        ],
    )

    return {
        "origin": origin,
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": (
            residence_type
        ),
        "avg_glucose_level": (
            avg_glucose_level
        ),
        "bmi": bmi,
        "smoking_status": (
            smoking_status
        ),
    }


def display_result(
    result,
):
    print(
        "\n=== Resultado ==="
    )

    print(
        f"{SCORE_LABEL}: "
        f"{result['score']:.4f}"
    )

    print(
        f"Threshold: "
        f"{result['threshold']:.4f}"
    )

    print(
        f"Clasificación: "
        f"{safe_classification_label(result['prediction'])}"
    )

    print(
        f"Modelo: "
        f"{result['model_version']}"
    )

    print(
        "\nAVISO:"
    )

    print(
        CLINICAL_DISCLAIMER
    )


def run():
    model_service = ModelService()

    try:
        print(
            "Cargando modelo..."
        )

        model_service.load()

        prediction_service = (
            PredictionService(
                model_service
            )
        )

        patient_data = (
            collect_patient_data()
        )

        request = PredictionRequest(
            **patient_data
        )

        result = (
            prediction_service.predict(
                request
            )
        )

        display_result(
            result
        )

    except ValidationError as exc:
        print(
            "\nLos datos introducidos "
            "no son válidos."
        )

        print(exc)

    except Exception as exc:
        print(
            "\nNo se pudo realizar "
            "la predicción."
        )

        print(
            f"Detalle: {exc}"
        )


if __name__ == "__main__":
    run()
