# Auditoría del dataset — Stroke Risk AI

## Objetivo

Evaluar la estructura y calidad inicial del dataset antes de realizar transformaciones o entrenar modelos de Machine Learning.

## Dimensiones

- Registros: **4.981**
- Variables: **11**
- Variable objetivo: `stroke`

## Variables

| Variable | Tipo |
|---|---|
| `gender` | Categórica |
| `age` | Numérica |
| `hypertension` | Binaria |
| `heart_disease` | Binaria |
| `ever_married` | Categórica |
| `work_type` | Categórica |
| `Residence_type` | Categórica |
| `avg_glucose_level` | Numérica |
| `bmi` | Numérica |
| `smoking_status` | Categórica |
| `stroke` | Binaria / Target |

## Calidad de los datos

### Valores nulos

No se han detectado valores nulos en ninguna de las 11 variables.

### Duplicados

No se han detectado filas completamente duplicadas.

### Variables categóricas

Las categorías observadas son coherentes con la estructura esperada del dataset.

Se ha identificado un aspecto relevante en `smoking_status`:

- `never smoked`: 1.838
- `Unknown`: 1.500
- `formerly smoked`: 867
- `smokes`: 776

La categoría `Unknown` representa información desconocida sobre el hábito tabáquico y afecta a una parte importante del dataset. Su tratamiento deberá decidirse durante el preprocessing y no durante esta auditoría.

### Rangos numéricos

| Variable | Mínimo | Media | Mediana | Máximo |
|---|---:|---:|---:|---:|
| `age` | 0.08 | 43.42 | 45.00 | 82.00 |
| `avg_glucose_level` | 55.12 | 105.94 | 91.85 | 271.74 |
| `bmi` | 14.00 | 28.50 | 28.10 | 48.90 |

No se han detectado valores imposibles mediante las reglas básicas de validación aplicadas:

- No existen edades negativas ni superiores a 120 años.
- No existen valores de glucosa menores o iguales a 0.
- No existen valores de BMI menores o iguales a 0.
- `hypertension`, `heart_disease` y `stroke` contienen únicamente valores `0` y `1`.

Los valores extremos observados no se eliminarán automáticamente. Deberán analizarse durante el EDA y preprocessing antes de decidir cualquier tratamiento.

## Distribución de la variable objetivo

La variable `stroke` presenta la siguiente distribución:

| Clase | Registros | Porcentaje |
|---|---:|---:|
| `0` — No stroke | 4.733 | 95,02 % |
| `1` — Stroke | 248 | 4,98 % |

**Ratio clase mayoritaria/minoritaria: 19,08:1.**

El dataset presenta, por tanto, un **fuerte desbalanceo de clases**.

Esta característica deberá tenerse en cuenta durante el entrenamiento y evaluación de los modelos. La accuracy por sí sola no será una métrica suficiente, ya que un modelo que favorezca sistemáticamente la clase mayoritaria podría obtener una accuracy elevada sin detectar adecuadamente pacientes de la clase positiva.

## Problemas y aspectos a tratar

1. **Fuerte desbalanceo de la variable objetivo `stroke`.**
2. **Alta presencia de `Unknown` en `smoking_status`.**
3. **Existencia de valores extremos en variables numéricas que deberán analizarse antes de decidir si requieren tratamiento.**
4. **Presencia de edades infantiles muy bajas**, que son válidas según las comprobaciones básicas pero deberán considerarse durante el análisis exploratorio.

## Conclusión

El dataset presenta una estructura consistente, sin valores nulos ni duplicados y sin valores imposibles detectados mediante las reglas básicas aplicadas.

El principal reto identificado es el fuerte desbalanceo de `stroke`, con solo un **4,98 % de casos positivos**. Este hecho condicionará posteriormente la selección de métricas, las estrategias de entrenamiento y el tratamiento del desbalanceo.