# Análisis de optimización de hiperparámetros

## Objetivo

Optimizar los hiperparámetros del modelo candidato seleccionado para mejorar su capacidad de detectar pacientes con `stroke` de forma reproducible.

## Modelo seleccionado

Se optimiza:

**Logistic Regression con `class_weight="balanced"`**

Este modelo fue seleccionado en las fases anteriores por su buen rendimiento, estabilidad y capacidad para reducir falsos negativos.

## Metodología

La optimización se realiza mediante **Optuna**.

Se ejecutan:

- 30 trials.
- Validación cruzada estratificada de 5 folds.
- Seed fija `42`.
- Optimización exclusivamente sobre el conjunto `train`.
- El conjunto `test` permanece completamente aislado.

## Hiperparámetros explorados

Optuna evalúa diferentes valores de:

- `C`: intensidad de regularización.
- `solver`: algoritmo utilizado para optimizar Logistic Regression.
- `max_iter`: número máximo de iteraciones.

## Métrica principal

La métrica utilizada como objetivo de optimización es **Recall**.

Se prioriza Recall porque en este problema es especialmente importante reducir los falsos negativos, es decir, evitar que pacientes con `stroke` sean clasificados como negativos.

También se registran:

- ROC-AUC.
- PR-AUC.

## Mejor configuración encontrada

| Parámetro | Resultado |
|---|---:|
| Recall CV | 0.8409 |
| C | 0.001486 |
| Solver | liblinear |
| Max iterations | 500 |
| ROC-AUC medio | 0.8285 |
| PR-AUC medio | 0.1870 |

## Comparación

Antes de la optimización, Logistic Regression con `class_weight="balanced"` obtuvo un Recall medio aproximado de:

`0.759`

Después de la optimización:

`0.841`

Por tanto, la optimización mejora la capacidad del modelo para detectar la clase positiva.

## Configuración candidata

La configuración seleccionada para las siguientes fases es:

`LogisticRegression(C=0.001486, solver="liblinear", max_iter=500, class_weight="balanced", random_state=42)`

## Conclusión

Optuna encuentra una configuración con mayor Recall medio que la configuración inicial.

La mejor configuración utiliza una regularización fuerte y el solver `liblinear`.

Esta configuración queda seleccionada como candidata para las siguientes fases del proyecto.

El conjunto `test` no ha participado en ningún momento en la búsqueda ni selección de hiperparámetros.