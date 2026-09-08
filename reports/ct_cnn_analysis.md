# Prototipo CNN para imágenes CT cerebrales

## Objetivo

Este experimento evalúa la viabilidad técnica de una CNN pequeña para clasificar
cortes de CT cerebral como `Normal` o `Stroke`. El prototipo constituye el
componente de imagen del proyecto multimodal y se mantiene completamente
separado del modelo tabular desplegado.

El objetivo es experimental. Sus resultados no representan una validación
clínica ni permiten utilizar el modelo como herramienta diagnóstica.

## Dataset

Se utilizó el **Brain Stroke CT Image Dataset de Afridi Rahman**, compuesto por
2.501 imágenes JPEG:

| Clase | Imágenes |
|---|---:|
| Normal | 1.551 |
| Stroke | 950 |
| **Total** | **2.501** |

Los nombres permiten identificar 82 grupos visibles: 51 `Normal` y 31
`Stroke`. El campo `group_id` se utiliza como proxy de serie o grupo de cortes;
no puede interpretarse como identificador de paciente porque el dataset no
incluye metadata suficiente para demostrarlo.

## Protocolo de split

El split se realizó a nivel de grupo, nunca por imagen individual. Cada grupo
completo pertenece exclusivamente a train, validation o test:

| Split | Grupos | Imágenes | Normal | Stroke | Proporción |
|---|---:|---:|---:|---:|---:|
| Train | 52 | 1.611 | 981 | 630 | 64,41% |
| Validation | 13 | 395 | 245 | 150 | 15,79% |
| Test | 17 | 495 | 325 | 170 | 19,79% |

La separación es aproximadamente 64/16/20, con seed 42 y estratificación por
clase a nivel de grupo. Se verificó que existe **cero group leakage** y **cero
image overlap** entre los tres splits.

Esta estrategia reduce el riesgo de que cortes relacionados de una misma serie
aparezcan simultáneamente en entrenamiento y evaluación.

## Pipeline de imágenes

Las imágenes se procesaron mediante `tf.data` con las siguientes operaciones:

- Decodificación JPEG en grayscale.
- Redimensionado a 224×224.
- Un único canal: `224×224×1`.
- Conversión a `float32`.
- Normalización al intervalo `[0, 1]`.
- Shuffle reproducible únicamente en train.
- Sin shuffle ni augmentation en validation y test.

El augmentation se aplicó exclusivamente durante entrenamiento y de forma
conservadora: rotaciones pequeñas, traslaciones de hasta el 3% y zoom leve de
±5%. No se utilizaron flips, deformaciones fuertes ni transformaciones de
color.

## Arquitectura

Se entrenó una única CNN predefinida, sin búsqueda de hiperparámetros:

```text
Input 224×224×1
→ Conv2D 16 + BatchNorm + ReLU + MaxPooling
→ Conv2D 32 + BatchNorm + ReLU + MaxPooling
→ Conv2D 64 + BatchNorm + ReLU + MaxPooling
→ GlobalAveragePooling
→ Dense 32 ReLU
→ Dropout 0.40
→ Dense 1 sigmoid
```

Las capas convolucionales y la capa densa de 32 unidades usan regularización L2
de `1e-4`. Global Average Pooling evita una capa `Flatten` de gran tamaño y
limita la complejidad del prototipo.

## Entrenamiento

- Optimizador: Adam.
- Learning rate inicial: `3e-4`.
- Loss: Binary Crossentropy.
- Batch size: 16.
- Class weights calculados exclusivamente con train.
- Early stopping sobre `val_loss`, restaurando los mejores pesos.
- `ReduceLROnPlateau` sobre `val_loss`.
- Mejor época: **18**.
- Épocas ejecutadas: **28**.
- Mejor validation loss: **0.615689**.

## Selección del threshold

El threshold se seleccionó exclusivamente con validation después de finalizar
el entrenamiento y restaurar el mejor checkpoint. El valor fijado fue:

```text
threshold = 0.37
```

El conjunto test no participó en la selección de arquitectura, pesos,
regularización, early stopping ni threshold. El valor 0.37 se mantuvo sin
cambios durante la evaluación final.

## Resultados

Las métricas son image-level y proceden del informe final generado por el
experimento:

| Split | Precision | Recall | F1 | ROC-AUC | PR-AUC | Specificity | Balanced accuracy | FN | FP | TP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 0.53694 | 0.81905 | 0.64865 | 0.77840 | 0.72364 | 0.54638 | 0.68271 | 114 | 445 | 516 | 536 |
| Validation | 0.48016 | 0.80667 | 0.60199 | 0.71804 | 0.65088 | 0.46531 | 0.63599 | 29 | 131 | 121 | 114 |
| Test | 0.31922 | 0.57647 | 0.41090 | 0.47633 | 0.38864 | 0.35692 | 0.46670 | 72 | 209 | 98 | 116 |

## Generalización y estabilidad

Validation mostró un rendimiento moderado, con recall 0.807, F1 0.602,
ROC-AUC 0.718 y PR-AUC 0.651. Sin embargo, el rendimiento disminuyó de forma
importante en los grupos de test no vistos:

| Métrica | Validation | Test | Diferencia test − validation |
|---|---:|---:|---:|
| Recall | 0.80667 | 0.57647 | −0.23020 |
| F1 | 0.60199 | 0.41090 | −0.19109 |
| ROC-AUC | 0.71804 | 0.47633 | −0.24172 |
| PR-AUC | 0.65088 | 0.38864 | −0.26224 |

El historial contiene 28 épocas. La mejor `val_loss` se alcanzó en la época 18
y early stopping restauró ese checkpoint. Después de esa época, la loss de
train continuó descendiendo mientras la loss y PR-AUC de validation empeoraron
al final del entrenamiento. Esto muestra tensión entre ajuste y generalización,
pero la caída final en test no debe atribuirse automáticamente a una única
causa de “overfitting clásico”.

Los resultados son compatibles con una **generalización limitada**. Entre las
explicaciones plausibles se encuentran el reducido número de grupos, posibles
diferencias de distribución entre grupos, correlación entre cortes y shortcuts
visuales relacionados con adquisición, encuadre o elementos no anatómicos.
Estas posibilidades no pueden distinguirse de forma causal con la información
disponible.

## Interpretación de ROC-AUC en test

El ROC-AUC de test es **0.47633**, inferior a 0.5. Esto representa un rendimiento
discriminativo insuficiente en los grupos de test no vistos: el modelo no
mantiene la capacidad de ordenación observada en train y validation.

Este resultado no debe corregirse reajustando el threshold después de observar
test. ROC-AUC evalúa el ranking de las probabilidades a través de todos los
thresholds, y cambiar a posteriori el threshold no resolvería la falta de
generalización discriminativa.

## Preservación del conjunto test

El modelo no se reajusta después de observar los resultados finales. Modificar
la arquitectura, los pesos, la regularización o el threshold en respuesta a
test convertiría ese conjunto en una extensión de validation e introduciría
test leakage.

Los resultados se conservan tal como fueron obtenidos para mantener la
independencia de la evaluación final. Cualquier experimento posterior deberá
definirse de antemano y evaluarse con nuevos datos independientes.

## Limitaciones

- Solo existen 82 grupos visibles y únicamente 52 participan en train.
- Los cortes de una misma serie están correlacionados y no son observaciones
  independientes.
- Los grupos visibles no equivalen necesariamente a pacientes independientes.
- El dataset usa JPEG y no conserva DICOM, unidades Hounsfield, windowing ni
  metadata de adquisición.
- Bordes, camilla, encuadre, contraste u otros elementos visuales pueden actuar
  como shortcuts.
- Las etiquetas parecen asignadas a nivel de serie; no todos los cortes de una
  serie `Stroke` tienen necesariamente una lesión visible.
- La evaluación es por imagen, por lo que las series con más cortes tienen más
  peso en las métricas.
- No existe validación externa, temporal ni por centro sanitario.
- No se dispone de información suficiente para interpretar clínicamente los
  errores o afirmar independencia por paciente.
- El prototipo no es apto para diagnóstico ni uso clínico.

## MLflow

El experimento quedó registrado de forma independiente del pipeline tabular:

- Experiment: `stroke-risk-ct-cnn`.
- Registered model: `stroke-risk-ct-cnn`.
- Model version: `1`.
- Stage: `prototype`.
- Deployed: `false`.

El modelo CT no sustituye ni modifica el modelo tabular utilizado por la
aplicación en producción.

## Conclusión

El prototipo demuestra técnicamente el pipeline CNN de imágenes solicitado:
split por grupos, procesamiento de CT, regularización, entrenamiento,
selección de threshold con validation, evaluación final protegida y registro
independiente en MLflow.

Los resultados de test, especialmente ROC-AUC 0.476, recall 0.576 y balanced
accuracy 0.467, **no justifican uso clínico ni integración en producción**. El
modelo tabular productivo no se sustituye.

El experimento también aporta un resultado metodológico importante: un split
estricto por grupos revela problemas de generalización que un split aleatorio
por imágenes podría ocultar al repartir cortes relacionados entre train y
evaluación.
