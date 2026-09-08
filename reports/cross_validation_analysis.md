# Análisis de validación cruzada

## Objetivo

Evaluar la estabilidad de los modelos candidatos mediante validación cruzada estratificada.

## Metodología

Se utiliza `StratifiedKFold` con:

- 5 folds.
- Shuffle activado.
- Seed fija `42`.
- Estratificación de la variable `stroke`.

La validación cruzada se realiza exclusivamente sobre el conjunto de entrenamiento.

Los conjuntos `validation` y `test` permanecen fuera del proceso.

## Resultados

| Modelo | ROC-AUC media | ROC-AUC std | PR-AUC media | PR-AUC std | Recall medio | F1 medio |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8326 | 0.0195 | 0.1919 | 0.0357 | 0.0000 | 0.0000 |
| Gradient Boosting | 0.8065 | 0.0305 | 0.1768 | 0.0479 | 0.0258 | 0.0471 |

## Logistic Regression

Logistic Regression presenta los mejores resultados medios tanto en ROC-AUC como en PR-AUC.

Además, las desviaciones estándar son relativamente bajas, especialmente en ROC-AUC, lo que indica que el comportamiento del modelo es estable entre los diferentes folds.

Aunque Recall y F1 son 0 con el threshold actual de 0.5, el ROC-AUC y PR-AUC muestran que el modelo sí tiene capacidad para diferenciar entre pacientes con mayor y menor riesgo.

## Gradient Boosting

Gradient Boosting obtiene resultados inferiores a Logistic Regression tanto en ROC-AUC como en PR-AUC.

Presenta además una dispersión mayor entre folds, especialmente en PR-AUC.

Consigue detectar algunos casos positivos con el threshold actual, pero el Recall medio sigue siendo muy bajo.

## Conclusión

La validación cruzada confirma a **Logistic Regression como el candidato principal** para las siguientes fases.

Su rendimiento es más alto y más estable que Gradient Boosting.

Los resultados también confirman que el principal problema no es únicamente la capacidad del modelo para ordenar el riesgo, sino la clasificación de la clase minoritaria con el threshold por defecto.

Las siguientes fases deberán estudiar el tratamiento del desbalanceo y posteriormente el ajuste del threshold.