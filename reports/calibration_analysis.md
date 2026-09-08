# Análisis de calibración de probabilidades

## Objetivo

El objetivo de esta fase es comprobar si las probabilidades generadas por el modelo pueden interpretarse de forma fiable.

Esto es especialmente importante si la aplicación muestra al usuario un **score o probabilidad de riesgo de stroke**, ya que una probabilidad mal calibrada podría resultar engañosa.

---

## Modelo evaluado

Se utiliza el modelo optimizado en las fases anteriores:

- **Modelo:** Logistic Regression
- **class_weight:** balanced
- **C:** 0.001486
- **solver:** liblinear
- **max_iter:** 500
- **random_state:** 42

La calibración se realiza utilizando únicamente los datos de entrenamiento y se evalúa sobre el conjunto de validación.

El conjunto de **test permanece completamente aislado** y no participa en esta decisión.

---

## Métodos comparados

Se compararon tres alternativas:

1. Modelo original sin calibración.
2. Calibración mediante `sigmoid`.
3. Calibración mediante `isotonic`.

Para evaluar la calidad de las probabilidades se utilizaron:

- **Calibration Curve**
- **Brier Score**

El Brier Score mide la diferencia entre las probabilidades predichas y los resultados reales.

Un valor más cercano a **0** indica una mejor calibración.

---

## Resultados

| Método | Brier Score |
|---|---:|
| **Sigmoid** | **0.043517** |
| Isotonic | 0.043753 |
| Sin calibrar | 0.195917 |

La calibración mejora claramente el Brier Score respecto al modelo original.

El método **sigmoid obtiene el mejor resultado**, aunque la diferencia respecto a isotonic es pequeña.

Por este motivo se selecciona **sigmoid** como método de calibración.

---

## Interpretación

El modelo sin calibrar obtiene un Brier Score de **0.195917**, mientras que después de aplicar calibración sigmoid el resultado baja hasta **0.043517**.

Esto supone una reducción aproximada del **78 % en el Brier Score**.

Por tanto, aunque el modelo original puede ser útil para ordenar o clasificar pacientes según su riesgo, sus probabilidades no deberían interpretarse directamente como porcentajes fiables de riesgo.

La calibración permite que los scores producidos por el modelo tengan una interpretación probabilística más adecuada.

---

## Interpretación del score para la aplicación

Si la aplicación muestra una probabilidad como:

`0.25`

debe interpretarse como una **estimación de riesgo generada por el modelo calibrado**, no como una certeza médica ni como un diagnóstico.

Por ejemplo, la interfaz podría expresarlo como:

> **Riesgo estimado por el modelo: 25 %**

Esta cifra representa la estimación estadística del modelo a partir de las variables disponibles.

No significa que el usuario vaya a sufrir un stroke ni sustituye la evaluación de un profesional sanitario.

---

## Decisión

Se selecciona la siguiente configuración para las probabilidades mostradas por el sistema:

- **Modelo:** Logistic Regression optimizada
- **Método de calibración:** sigmoid
- **Brier Score en validación:** 0.043517

La versión calibrada será la utilizada cuando sea necesario mostrar scores probabilísticos al usuario.

---

## Control de leakage

Durante esta fase:

- El modelo utiliza `train` para el entrenamiento.
- La comparación de calibración se realiza con `validation`.
- El conjunto `test` no se utiliza.
- No se selecciona ningún método utilizando información del conjunto de test.

El conjunto de test continúa reservado para la evaluación final del sistema.

---

## Conclusión

La evaluación demuestra que las probabilidades del modelo original presentan una calibración considerablemente peor que las obtenidas después de aplicar técnicas específicas de calibración.

La calibración **sigmoid** obtiene el mejor Brier Score (`0.043517`) y se selecciona como método definitivo.

De esta forma, si la aplicación muestra un score de riesgo, este procederá del modelo calibrado y deberá presentarse siempre como una **estimación estadística de riesgo**, no como un diagnóstico médico.