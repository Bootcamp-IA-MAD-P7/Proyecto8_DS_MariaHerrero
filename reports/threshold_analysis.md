# Análisis de optimización del threshold

## Objetivo

Seleccionar un threshold de decisión adecuado para el modelo optimizado, buscando mantener una alta capacidad de detección de pacientes con `stroke` y controlar los falsos negativos y falsos positivos.

## Modelo utilizado

Se utiliza el modelo seleccionado y optimizado en las fases anteriores:

**Logistic Regression con `class_weight="balanced"`**

Configuración:

- `C = 0.001486`
- `solver = "liblinear"`
- `max_iter = 500`
- `random_state = 42`

Versión del modelo:

`logreg_v1`

## Metodología

El modelo se entrena exclusivamente con el conjunto `train`.

La selección del threshold se realiza utilizando únicamente el conjunto `validation`.

Se evalúan thresholds entre:

`0.10` y `0.90`

Para cada threshold se calculan:

- Precision.
- Recall.
- F1.
- Falsos negativos.
- Falsos positivos.

También se genera la curva Precision-Recall.

El conjunto `test` permanece completamente aislado.

## Criterio de selección

Dado que el objetivo principal es detectar pacientes con riesgo de `stroke`, se establece como requisito:

**Recall >= 0.80**

Entre los thresholds que cumplen este requisito se selecciona el que produce el menor número de falsos positivos.

En caso de empate se utiliza F1 como criterio adicional.

## Threshold seleccionado

| Métrica | Resultado |
|---|---:|
| Threshold | 0.50 |
| Recall | 0.825 |
| Precision | 0.128 |
| F1 | 0.221 |
| Falsos negativos | 7 |
| Falsos positivos | 225 |

Con este threshold, el modelo detecta 33 de los 40 casos positivos presentes en validation.

## Interpretación

El threshold seleccionado es `0.50`.

Aunque coincide con el threshold habitual utilizado por defecto en clasificación binaria, en este proyecto no se utiliza simplemente por ser el valor estándar.

El valor ha sido comparado con otros thresholds y seleccionado explícitamente utilizando el conjunto de validation y el criterio definido para el proyecto.

Reducir más el threshold permitiría aumentar el Recall, pero también aumentaría el número de falsos positivos.

Aumentarlo reduciría las falsas alarmas, pero provocaría una pérdida de Recall superior al límite establecido.

## Versionado

El threshold queda asociado a:

`logreg_v1`

Threshold:

`0.50`

Esto permite mantener registrada la regla de decisión utilizada junto con la versión del modelo.

## Conclusión

Se selecciona un threshold de **0.50** para `logreg_v1`.

Esta configuración mantiene un Recall del **82.5 %**, con **7 falsos negativos** en validation.

La selección se ha realizado sin utilizar el conjunto `test`, que permanece reservado para la evaluación final del modelo.