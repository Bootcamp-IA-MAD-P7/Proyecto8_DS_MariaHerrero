# Análisis de explicabilidad del modelo

## Objetivo

El objetivo de esta fase es proporcionar explicaciones comprensibles sobre el comportamiento del modelo de predicción de stroke.

La explicabilidad se aborda en dos niveles:

- **Explicabilidad global:** identificar qué variables influyen más en el comportamiento general del modelo.
- **Explicabilidad individual:** identificar qué variables aumentan o disminuyen el score generado para una observación concreta.

Las explicaciones describen exclusivamente el comportamiento del modelo.

No deben interpretarse como relaciones causales ni como conclusiones médicas.

---

## Modelo explicado

Las explicaciones se generan sobre el modelo final:

- **Modelo:** Logistic Regression
- **Versión:** `logreg_v1`
- **Calibración:** sigmoid
- **Threshold:** 0.05

De esta forma, la explicación corresponde al mismo modelo utilizado para generar las predicciones finales.

---

## Explicabilidad global

Para analizar la importancia global de las variables se utiliza:

**Permutation Importance**

Este método mide cuánto empeora el rendimiento del modelo cuando los valores de una variable se mezclan aleatoriamente.

La métrica utilizada para medir ese cambio es:

`ROC-AUC`

Una importancia mayor indica que el modelo depende más de esa variable para mantener su capacidad predictiva.

Esto no significa que la variable sea una causa médica de stroke.

---

## Resultados globales

| Variable | Importancia media | Desviación |
|---|---:|---:|
| age | 0.250852 | 0.032246 |
| avg_glucose_level | 0.012910 | 0.010003 |
| hypertension | 0.004171 | 0.001481 |
| heart_disease | 0.001384 | 0.000733 |
| ever_married | 0.001374 | 0.003571 |
| smoking_status | 0.000869 | 0.002871 |
| Residence_type | 0.000063 | 0.000240 |
| bmi | -0.000284 | 0.002820 |
| work_type | -0.000816 | 0.001395 |
| gender | -0.001291 | 0.001127 |

---

## Interpretación global

La variable con mayor influencia global sobre el comportamiento del modelo es:

**age**

Su importancia es considerablemente superior a la del resto de variables.

Después aparecen variables como:

- `avg_glucose_level`
- `hypertension`
- `heart_disease`

con una influencia global menor.

Los valores de importancia cercanos a cero indican que alterar esa variable tiene poco impacto sobre el ROC-AUC del modelo en el conjunto evaluado.

Las pequeñas importancias negativas observadas en algunas variables no deben interpretarse como un efecto protector.

Pueden aparecer debido a variabilidad estadística o a que esa variable aporta muy poca información adicional al modelo.

---

## Explicabilidad individual

Para explicar una predicción individual se compara el score original de la persona con el score obtenido al sustituir cada variable, una a una, por un valor de referencia.

Los valores de referencia se calculan utilizando exclusivamente los datos de entrenamiento:

- Variables numéricas: mediana.
- Variables categóricas: moda.

La diferencia entre ambos scores se utiliza como medida de influencia.

---

## Interpretación de la influencia individual

Una influencia positiva significa:

> El valor actual de esa variable aumenta el score generado por el modelo respecto al valor de referencia.

Una influencia negativa significa:

> El valor actual de esa variable disminuye el score generado por el modelo respecto al valor de referencia.

Estas influencias describen exclusivamente el comportamiento matemático del modelo.

No representan causalidad médica.

---

## Ejemplo individual

Para una observación de validación se obtuvo:

- **Score:** 0.033565
- **Threshold:** 0.05
- **Predicción:** 0

Como el score es inferior al threshold seleccionado, el modelo clasifica la observación como:

`No stroke / riesgo por debajo del threshold`

---

## Factores que aumentan el score

### age

- Valor observado: `53`
- Valor de referencia: `45`
- Influencia: `+0.009755`

Para este registro, el valor de `age` aumenta el score generado por el modelo respecto al valor de referencia.

### bmi

- Valor observado: `35.4`
- Valor de referencia: `28.3`
- Influencia: `+0.004219`

Para este registro, el valor de `bmi` aumenta el score generado por el modelo respecto al valor de referencia.

### Residence_type

- Valor observado: `Rural`
- Valor de referencia: `Urban`
- Influencia: `+0.000545`

Su influencia sobre el score es pequeña.

---

## Factores que disminuyen el score

### gender

- Valor observado: `Male`
- Valor de referencia: `Female`
- Influencia: `-0.001985`

### avg_glucose_level

- Valor observado: `90.12`
- Valor de referencia: `92.04`
- Influencia: `-0.000540`

### smoking_status

- Valor observado: `Unknown`
- Valor de referencia: `never smoked`
- Influencia: `-0.000482`

Estos valores reducen ligeramente el score del modelo respecto a sus respectivos valores de referencia.

---

## Importante: ausencia de causalidad

Las explicaciones generadas no indican que una variable:

- cause un stroke;
- prevenga un stroke;
- produzca directamente un cambio clínico.

Únicamente indican cómo el modelo utiliza las variables para modificar su score.

Por tanto, expresiones como:

`La edad causa un aumento del riesgo`

deben evitarse.

La formulación correcta sería:

`El valor de edad de este registro aumenta el score estimado por el modelo respecto al valor de referencia.`

---

## Formato para FastAPI

La explicación individual está preparada para ser consumida directamente por una API.

La función:

`explanation_for_api()`

devuelve una estructura JSON compatible con FastAPI.

Ejemplo:

```json
{
  "model_version": "logreg_v1",
  "score": 0.033565,
  "threshold": 0.05,
  "prediction": 0,
  "factors_increasing_score": [],
  "factors_decreasing_score": [],
  "interpretation": "Las influencias describen cómo cada variable modifica el score generado por el modelo respecto a un valor de referencia.",
  "disclaimer": "La explicación describe el comportamiento del modelo y no implica causalidad médica ni constituye un diagnóstico."
}