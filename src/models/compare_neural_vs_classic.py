"""Compare stored test results for the final classic and neural models.

This module never loads a dataset or model. It only summarizes metrics that
were already produced by the final, protected test evaluations.
"""

from pathlib import Path

import pandas as pd


CLASSIC_METRICS_PATH = Path("reports/final_model_metrics.csv")
CLASSIC_ANALYSIS_PATH = Path("reports/final_model_analysis.md")
NEURAL_METRICS_PATH = Path("reports/tabular_nn_metrics.csv")
NEURAL_HISTORY_PATH = Path("reports/tabular_nn_training_history.csv")
COMPARISON_PATH = Path("reports/neural_vs_classic_comparison.csv")
ANALYSIS_PATH = Path("reports/neural_vs_classic_analysis.md")

MODEL_CLASSIC = "Classic ML"
MODEL_NEURAL = "Neural Network"

METRICS = (
    ("Precision", "precision", True),
    ("Recall", "recall", True),
    ("F1", "f1", True),
    ("ROC-AUC", "roc_auc", True),
    ("PR-AUC", "pr_auc", True),
    ("False negatives", "false_negatives", False),
    ("False positives", "false_positives", False),
    ("True positives", "true_positives", True),
    ("True negatives", "true_negatives", True),
)


def load_metrics(path):
    """Load the single stored test row from a metrics report."""
    report = pd.read_csv(path)
    split_column = "dataset" if "dataset" in report.columns else "split"
    test_rows = report[report[split_column] == "test"]
    if len(test_rows) != 1:
        raise ValueError(f"Expected exactly one test row in {path}.")
    return test_rows.iloc[0]


def better_model(classic_value, neural_value, higher_is_better):
    """Return the better model, accounting for metric direction."""
    if classic_value == neural_value:
        return "Tie"
    neural_is_better = neural_value > classic_value
    if not higher_is_better:
        neural_is_better = not neural_is_better
    return MODEL_NEURAL if neural_is_better else MODEL_CLASSIC


def build_comparison(classic, neural):
    """Build the comparison from two previously stored test rows."""
    rows = []
    for label, column, higher_is_better in METRICS:
        classic_value = classic[column]
        neural_value = neural[column]
        if column.startswith(("false_", "true_")):
            classic_value = int(classic_value)
            neural_value = int(neural_value)
        rows.append(
            {
                "Metric": label,
                MODEL_CLASSIC: classic_value,
                MODEL_NEURAL: neural_value,
                "Difference": neural_value - classic_value,
                "Better": better_model(
                    classic_value,
                    neural_value,
                    higher_is_better,
                ),
            }
        )
    return pd.DataFrame(rows)


def overfitting_summary(classic_report, neural_report, history):
    """Summarize generalization using only stored report values."""
    classic = classic_report.set_index("dataset")
    neural = neural_report.set_index("dataset")
    best_epoch = int(neural.loc["test", "best_epoch"])
    best_history = history.loc[history["epoch"] == best_epoch].iloc[0]
    final_history = history.iloc[-1]
    return {
        "classic_f1_gap": abs(
            classic.loc["train", "f1"] - classic.loc["test", "f1"]
        ),
        "classic_roc_auc_gap": abs(
            classic.loc["train", "roc_auc"] - classic.loc["test", "roc_auc"]
        ),
        "classic_pr_auc_gap": abs(
            classic.loc["train", "pr_auc"] - classic.loc["test", "pr_auc"]
        ),
        "neural_train_validation_f1_gap": neural.loc[
            "test", "train_validation_f1_gap"
        ],
        "neural_train_validation_pr_auc_gap": neural.loc[
            "test", "train_validation_pr_auc_gap"
        ],
        "best_epoch": best_epoch,
        "epochs_executed": int(neural.loc["test", "epochs_executed"]),
        "best_val_loss": best_history["val_loss"],
        "final_train_loss": final_history["loss"],
        "final_val_loss": final_history["val_loss"],
        "best_val_pr_auc": best_history["val_pr_auc"],
        "final_val_pr_auc": final_history["val_pr_auc"],
    }


def format_markdown_table(comparison):
    header = "| Metric | Classic ML | Neural Network | Difference | Better |"
    separator = "|---|---:|---:|---:|---|"
    rows = []
    count_metrics = {
        "False negatives",
        "False positives",
        "True positives",
        "True negatives",
    }
    for row in comparison.to_dict("records"):
        metric = row["Metric"]
        if metric in count_metrics:
            values = [
                str(int(row[key]))
                for key in (MODEL_CLASSIC, MODEL_NEURAL, "Difference")
            ]
        else:
            values = [
                f"{float(row[key]):.6f}"
                for key in (MODEL_CLASSIC, MODEL_NEURAL, "Difference")
            ]
        rows.append(
            f"| {metric} | {values[0]} | {values[1]} | "
            f"{values[2]} | {row['Better']} |"
        )
    return "\n".join([header, separator, *rows])


def build_analysis(comparison, overfitting, classic_analysis):
    if "test se utiliza únicamente para la evaluación final" not in classic_analysis:
        raise ValueError("Classic analysis does not document protected final test use.")

    table = format_markdown_table(comparison)
    return f"""# Comparación: red neuronal tabular vs ML clásico

## Objetivo

Comparar objetivamente la Logistic Regression final (`logreg_v1`) y la red
neuronal tabular (`tabular_nn_v1`) para determinar cuál responde mejor al
objetivo de cribado de riesgo de ictus. Esta comparación reutiliza resultados
finales ya almacenados; no reentrena modelos ni vuelve a evaluar test.

## Protocolo común y legitimidad

Ambos modelos usan el mismo dataset, los mismos splits estratificados de train,
validation y test (seed 42), el target `stroke`, el mismo preprocessing y las
mismas definiciones de métricas. El conjunto test permaneció protegido hasta
la evaluación final. La Logistic Regression está calibrada con sigmoid y usa
threshold 0.05; la red neuronal no está calibrada y usa threshold 0.50. Por
ello, los valores numéricos de ambos thresholds no son comparables entre sí.

## Resultados finales en test

La diferencia se define como `Neural Network - Classic ML`. Para FN y FP un
valor menor es mejor; para el resto, un valor mayor es mejor.

{table}

## Análisis por métrica

- **Recall:** aumenta de 0.780000 a 0.840000 (+0.060000), una mejora relevante
  para un sistema de cribado.
- **Precision:** pasa de 0.137809 a 0.138158 (+0.000349); es un empate práctico.
- **F1:** mejora ligeramente de 0.234234 a 0.237288 (+0.003054).
- **ROC-AUC:** aumenta de 0.825723 a 0.835333 (+0.009609).
- **PR-AUC:** aumenta de 0.145957 a 0.163629 (+0.017672), especialmente
  informativo por el fuerte desbalanceo de clases.
- **Falsos negativos:** disminuyen de 11 a 8; la NN detecta tres casos positivos
  adicionales.
- **Falsos positivos:** aumentan de 244 a 262, es decir, 18 alertas adicionales.

## Overfitting y estabilidad

La Logistic Regression presenta gaps train-test reducidos: F1
{overfitting['classic_f1_gap']:.6f}, ROC-AUC
{overfitting['classic_roc_auc_gap']:.6f} y PR-AUC
{overfitting['classic_pr_auc_gap']:.6f}. Esto respalda una mayor estabilidad.

La NN presenta gaps train-validation de F1
{overfitting['neural_train_validation_f1_gap']:.6f} y PR-AUC
{overfitting['neural_train_validation_pr_auc_gap']:.6f}. Su mejor época fue la
{overfitting['best_epoch']} y ejecutó {overfitting['epochs_executed']} épocas.
Después del mínimo de validation loss ({overfitting['best_val_loss']:.6f}), la
pérdida de validation terminó en {overfitting['final_val_loss']:.6f}, mientras
la pérdida de train terminó en {overfitting['final_train_loss']:.6f}; además,
validation PR-AUC pasó de {overfitting['best_val_pr_auc']:.6f} en la mejor época
a {overfitting['final_val_pr_auc']:.6f}. Esto muestra overfitting leve, limitado
mediante early stopping con restauración del mejor checkpoint.

## Trade-offs y conclusión

La **red neuronal tabular es la ganadora experimental** para el objetivo de
cribado: mejora recall y PR-AUC, reduce los falsos negativos y también supera
ligeramente al modelo clásico en F1 y ROC-AUC. Sin embargo, genera 18 falsos
positivos adicionales, la precision está prácticamente empatada, muestra algo
más de overfitting, es más compleja y sus probabilidades no están calibradas.

Esta conclusión no implica sustituir la Logistic Regression desplegada. El
modelo clásico conserva ventajas de estabilidad, interpretabilidad, calibración
y sencillez operativa. No existe una prueba de significancia estadística que
permita afirmar que las mejoras pequeñas de la NN sean concluyentes.

## Limitaciones

- La comparación contiene una única evaluación final sobre el split test.
- No se calcularon intervalos de confianza ni pruebas de significancia.
- El análisis clásico de estabilidad usa train-test, mientras que la NN aporta
  además curvas train-validation por época.
- Los modelos aplican estrategias y thresholds distintos, seleccionados sobre
  validation según su propio protocolo.
- Los resultados apoyan una comparación experimental, no una conclusión
  diagnóstica ni una decisión automática de despliegue.
"""


def generate_reports(
    classic_metrics_path=CLASSIC_METRICS_PATH,
    classic_analysis_path=CLASSIC_ANALYSIS_PATH,
    neural_metrics_path=NEURAL_METRICS_PATH,
    neural_history_path=NEURAL_HISTORY_PATH,
    comparison_path=COMPARISON_PATH,
    analysis_path=ANALYSIS_PATH,
):
    classic_report = pd.read_csv(classic_metrics_path)
    neural_report = pd.read_csv(neural_metrics_path)
    history = pd.read_csv(neural_history_path)
    classic_analysis = Path(classic_analysis_path).read_text(encoding="utf-8")

    comparison = build_comparison(
        load_metrics(classic_metrics_path),
        load_metrics(neural_metrics_path),
    )
    overfitting = overfitting_summary(classic_report, neural_report, history)

    comparison_path = Path(comparison_path)
    analysis_path = Path(analysis_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    analysis_path.write_text(
        build_analysis(comparison, overfitting, classic_analysis),
        encoding="utf-8",
    )
    return comparison


def main():
    generate_reports()


if __name__ == "__main__":
    main()
