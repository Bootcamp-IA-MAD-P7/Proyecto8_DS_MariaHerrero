# Selección y evaluación del modelo final

## Objetivo

El objetivo de esta fase es seleccionar el modelo definitivo del proyecto y realizar su evaluación final utilizando el conjunto de test.

Hasta esta fase, el conjunto `test` se mantuvo aislado y no participó en:

- selección de modelos;
- validación cruzada;
- tratamiento del desbalanceo;
- optimización de hiperparámetros;
- calibración;
- selección del threshold.

El conjunto de test se utiliza únicamente para la evaluación final.

---

## Modelo seleccionado

El modelo final seleccionado es:

- **Modelo:** Logistic Regression
- **class_weight:** balanced
- **C:** 0.001486
- **solver:** liblinear
- **max_iter:** 500
- **random_state:** 42
- **Calibración:** sigmoid
- **Threshold final:** 0.05
- **Versión:** logreg_v1

---

## Orden del pipeline final

El flujo definitivo del modelo es:

`Preprocessing → Logistic Regression → Calibración sigmoid → Threshold 0.05`

El threshold de `0.05` fue seleccionado sobre el conjunto de validación después de aplicar la calibración.

Este paso fue necesario porque la calibración modifica la escala de las probabilidades y, por tanto, el threshold previamente seleccionado sobre probabilidades no calibradas no podía reutilizarse directamente.

---

## Justificación de la selección

Durante el proyecto se compararon distintos modelos clásicos:

- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM
- Decision Tree

Logistic Regression fue seleccionada como modelo final por ofrecer el mejor equilibrio entre:

- capacidad de detección de la clase positiva;
- estabilidad en validación cruzada;
- menor riesgo de sobreajuste;
- interpretabilidad;
- posibilidad de calibrar las probabilidades;
- comportamiento adecuado en un problema con fuerte desbalanceo.

Otros modelos presentaron problemas como menor recall o mayor sobreajuste.

---

## Calibración

Las probabilidades del modelo original no estaban suficientemente calibradas.

Se compararon:

- modelo sin calibrar;
- calibración sigmoid;
- calibración isotonic.

Resultados:

| Método | Brier Score |
|---|---:|
| Sigmoid | 0.043517 |
| Isotonic | 0.043753 |
| Sin calibrar | 0.195917 |

Se seleccionó la calibración **sigmoid**.

---

## Threshold final

Después de calibrar el modelo se volvió a optimizar el threshold utilizando exclusivamente el conjunto de validación.

El criterio utilizado fue:

`Recall >= 0.80`

Entre los thresholds que cumplían este requisito se seleccionó el que producía menos falsos positivos.

El threshold seleccionado fue:

`0.05`

Resultados en validación:

| Métrica | Resultado |
|---|---:|
| Threshold | 0.05 |
| Recall | 0.80 |
| Precision | 0.141 |
| F1 | 0.240 |
| Falsos negativos | 8 |
| Falsos positivos | 195 |

---

## Resultados finales en train

| Métrica | Resultado |
|---|---:|
| Precision | 0.1299 |
| Recall | 0.8165 |
| F1 | 0.2242 |
| ROC-AUC | 0.8345 |
| PR-AUC | 0.1807 |
| Falsos negativos | 29 |
| Falsos positivos | 864 |
| Verdaderos positivos | 129 |
| Verdaderos negativos | 2165 |

---

## Resultados finales en test

| Métrica | Resultado |
|---|---:|
| Precision | 0.1378 |
| Recall | 0.7800 |
| F1 | 0.2342 |
| ROC-AUC | 0.8257 |
| PR-AUC | 0.1460 |
| Falsos negativos | 11 |
| Falsos positivos | 244 |
| Verdaderos positivos | 39 |
| Verdaderos negativos | 703 |

El modelo detecta:

**39 de los 50 casos de stroke presentes en test.**

Esto supone un recall del:

**78 %**

---

## Matriz de confusión final

En test se obtienen:

| | Predicción No Stroke | Predicción Stroke |
|---|---:|---:|
| Real No Stroke | 703 | 244 |
| Real Stroke | 11 | 39 |

El modelo prioriza la detección de stroke, reduciendo los falsos negativos a costa de aumentar los falsos positivos.

Este comportamiento es coherente con el objetivo del proyecto, donde no detectar un posible caso de stroke se considera especialmente relevante.

---

## Comparación train/test

El F1 obtenido es:

- **Train F1:** 0.2242
- **Test F1:** 0.2342

Gap absoluto:

`0.0101`

Esto equivale aproximadamente a **1,01 puntos porcentuales**.

El requisito establecido era mantener el gap por debajo de 5 puntos porcentuales.

Por tanto:

**El requisito de generalización se cumple.**

También se observa estabilidad en ROC-AUC:

- Train: 0.8345
- Test: 0.8257

La diferencia entre ambos conjuntos es pequeña.

---

## Interpretación de los resultados

El modelo presenta una precision relativamente baja debido al fuerte desbalanceo del dataset y al threshold seleccionado para priorizar recall.

Esto implica que se generan falsos positivos.

Sin embargo, el objetivo del sistema no es realizar un diagnóstico médico, sino identificar casos que puedan presentar un riesgo elevado según el modelo.

Por ello, el resultado debe utilizarse como una señal de apoyo o screening y no como una decisión clínica definitiva.

---

## Artefactos versionados

La versión final genera y guarda:

- modelo calibrado;
- preprocessing entrenado;
- threshold;
- metadatos asociados a la versión.

Versión:

`logreg_v1`

Threshold:

`0.05`

Método de calibración:

`sigmoid`

El threshold queda asociado explícitamente a la versión del modelo y fue seleccionado sobre validation.

---

## Control de leakage

El conjunto de test permaneció aislado durante todas las decisiones de modelado.

El orden seguido fue:

`train → validation → selección final → test`

El test se utilizó únicamente después de fijar:

- modelo;
- hiperparámetros;
- estrategia de desbalanceo;
- calibración;
- threshold.

Los resultados obtenidos en test no se utilizaron para modificar posteriormente ninguna de estas decisiones.

---

## Conclusión

Se selecciona como modelo final **Logistic Regression calibrada con sigmoid**, utilizando un threshold de **0.05**.

En test alcanza:

- Recall: **78 %**
- ROC-AUC: **0.826**
- F1: **0.234**
- Falsos negativos: **11**
- Verdaderos positivos: **39**

El gap de F1 entre train y test es de aproximadamente **1,01 puntos porcentuales**, por lo que cumple el requisito máximo del 5 %.

El modelo final queda versionado como:

`logreg_v1`