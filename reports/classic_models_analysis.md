# Comparación de modelos clásicos

## Objetivo

Comparar varias familias de modelos de Machine Learning utilizando el mismo protocolo de entrenamiento y validación para identificar candidatos adecuados para las siguientes fases de optimización.

## Protocolo

Todos los modelos:

- utilizan el mismo preprocessing;
- se entrenan exclusivamente con `train`;
- se evalúan sobre `validation`;
- mantienen el conjunto `test` completamente aislado;
- utilizan las mismas métricas.

## Resultados

| Modelo | Precision | Recall | F1 | ROC-AUC | PR-AUC | FN | Train F1 | Validation F1 | F1 Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.000 | 0.000 | 0.000 | 0.843 | 0.231 | 40 | 0.000 | 0.000 | 0.000 |
| Random Forest | 0.000 | 0.000 | 0.000 | 0.792 | 0.194 | 40 | 1.000 | 0.000 | 1.000 |
| Gradient Boosting | 0.000 | 0.000 | 0.000 | 0.834 | 0.181 | 40 | 0.328 | 0.000 | 0.328 |
| SVM | 0.000 | 0.000 | 0.000 | 0.693 | 0.164 | 40 | 0.025 | 0.000 | 0.025 |
| Decision Tree | 0.184 | 0.175 | 0.179 | 0.567 | 0.074 | 33 | 1.000 | 0.179 | 0.821 |

## Análisis de overfitting

### Random Forest

Presenta un `Train F1 = 1.0` y `Validation F1 = 0.0`.

Esto indica un sobreajuste muy fuerte: el modelo aprende perfectamente los datos de entrenamiento pero no generaliza adecuadamente sobre validation.

### Decision Tree

Presenta un `Train F1 = 1.0` frente a `Validation F1 = 0.179`.

También muestra un sobreajuste elevado, con un gap aproximado de `0.821`.

### Gradient Boosting

El gap es menor que en Random Forest y Decision Tree, pero sigue existiendo diferencia entre train y validation.

### Logistic Regression

No presenta gap de F1 porque con el threshold actual de `0.5` no clasifica ningún caso positivo ni en train ni en validation.

Sin embargo, presenta los mejores valores de `ROC-AUC` y `PR-AUC`, lo que indica que el modelo sí tiene capacidad para ordenar pacientes según riesgo.

### SVM

Presenta poco gap entre train y validation, pero su capacidad de discriminación es inferior a Logistic Regression y Gradient Boosting.

## Candidatos para fases posteriores

Los candidatos más prometedores son:

1. **Logistic Regression**
   - Mejor ROC-AUC.
   - Mejor PR-AUC.
   - Modelo sencillo e interpretable.
   - Necesita tratamiento del desbalanceo y/o ajuste de threshold.

2. **Gradient Boosting**
   - ROC-AUC competitivo.
   - Puede capturar relaciones no lineales.
   - Requiere controlar el overfitting y mejorar la detección de la clase positiva.

Random Forest y Decision Tree no se descartan definitivamente, pero sus configuraciones actuales muestran sobreajuste elevado.

## Conclusión

Ningún modelo con la configuración actual resuelve adecuadamente la detección de la clase positiva.

La comparación confirma que el problema principal sigue siendo el fuerte desbalanceo del dataset y el uso del threshold por defecto.

Las siguientes fases deberán centrarse en validación cruzada, tratamiento del desbalanceo, optimización de hiperparámetros y ajuste del threshold.