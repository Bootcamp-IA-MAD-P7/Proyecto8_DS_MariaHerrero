# Análisis del tratamiento del desbalanceo

## Objetivo

Comparar diferentes estrategias para tratar el fuerte desbalanceo de la variable objetivo `stroke`, buscando mejorar especialmente la detección de pacientes de la clase positiva y reducir los falsos negativos.

## Estrategias evaluadas

Se compararon cuatro configuraciones:

- Baseline sin tratamiento del desbalanceo.
- Logistic Regression con `class_weight="balanced"`.
- Random Oversampling.
- SMOTE.

El resampling se aplica exclusivamente sobre los datos de entrenamiento. Los conjuntos de validación y test no son modificados.

## Resultados sobre validation

| Estrategia | Precision | Recall | F1 | ROC-AUC | PR-AUC | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.000 | 0.000 | 0.000 | 0.843 | 0.231 | 40 | 0 |
| Class Weight | 0.156 | 0.825 | 0.263 | 0.843 | 0.229 | 7 | 178 |
| Random Oversampling | 0.157 | 0.825 | 0.264 | 0.842 | 0.233 | 7 | 177 |
| SMOTE | 0.163 | 0.825 | 0.272 | 0.840 | 0.208 | 7 | 170 |

Las tres técnicas de tratamiento del desbalanceo aumentan el Recall desde 0 hasta 0.825 y reducen los falsos negativos de 40 a 7.

En esta partición concreta, SMOTE obtiene el mejor F1 y el menor número de falsos positivos.

## Validación cruzada

Para comprobar si estos resultados se mantienen en diferentes particiones de los datos, se realiza validación cruzada estratificada sobre el conjunto de entrenamiento.

| Estrategia | Recall medio | Recall std | Precision media | F1 medio | ROC-AUC medio | PR-AUC medio | FN medios | FP medios |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Class Weight | 0.759 | 0.075 | 0.128 | 0.219 | 0.830 | 0.194 | 7.6 | 164.4 |
| Random Oversampling | 0.753 | 0.067 | 0.128 | 0.219 | 0.830 | 0.193 | 7.8 | 162.6 |
| SMOTE | 0.734 | 0.084 | 0.126 | 0.214 | 0.825 | 0.188 | 8.4 | 162.0 |

## Estrategia seleccionada

Se selecciona **Logistic Regression con `class_weight="balanced"`** como estrategia principal para las siguientes fases.

La decisión se basa principalmente en:

- Mayor Recall medio: `0.759`.
- Menor número medio de falsos negativos: `7.6`.
- ROC-AUC medio ligeramente superior.
- PR-AUC medio ligeramente superior.
- No requiere generar observaciones sintéticas.
- Mantiene un pipeline más sencillo.

Random Oversampling obtiene resultados muy similares y se mantiene como alternativa.

SMOTE, aunque obtiene buenos resultados en el validation original, presenta peores resultados medios al repetir el experimento mediante validación cruzada.

## Conclusión

El tratamiento del desbalanceo mejora de forma muy importante la capacidad del modelo para detectar la clase positiva.

El baseline no detectaba ninguno de los 40 casos positivos de validation, mientras que las estrategias de balanceo detectan 33 de los 40 casos.

`class_weight="balanced"` se utilizará como estrategia principal en las siguientes fases de optimización.

El conjunto `test` permanece completamente aislado y no se ha utilizado para seleccionar la estrategia.
