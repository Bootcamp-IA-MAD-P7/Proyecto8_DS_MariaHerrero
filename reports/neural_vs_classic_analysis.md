# Comparación: red neuronal tabular vs ML clásico

## Objetivo

Comparar objetivamente la Logistic Regression final (`logreg_v1`) y la red
neuronal tabular (`tabular_nn_v1`) para determinar cuál responde mejor al
objetivo de cribado de riesgo de ictus. Esta comparación reutiliza resultados
finales ya almacenados; no reentrena modelos ni vuelve a evaluar test.

## Protocolo común y legitimidad

Ambos modelos usan el mismo dataset, los mismos splits estratificados de train,
validation y test (seed 42), el target `stroke`, el mismo preprocessing y las
mismas definiciones de métricas. El conjunto test permaneció protegido hasta
la evaluación final. La Logistic Regression está calibrada con sigmoid y usa
threshold 0.05; la red neuronal no está calibrada y usa threshold 0.50. Por
ello, los valores numéricos de ambos thresholds no son comparables entre sí.

## Resultados finales en test

La diferencia se define como `Neural Network - Classic ML`. Para FN y FP un
valor menor es mejor; para el resto, un valor mayor es mejor.

| Metric | Classic ML | Neural Network | Difference | Better |
|---|---:|---:|---:|---|
| Precision | 0.137809 | 0.138158 | 0.000349 | Neural Network |
| Recall | 0.780000 | 0.840000 | 0.060000 | Neural Network |
| F1 | 0.234234 | 0.237288 | 0.003054 | Neural Network |
| ROC-AUC | 0.825723 | 0.835333 | 0.009609 | Neural Network |
| PR-AUC | 0.145957 | 0.163629 | 0.017672 | Neural Network |
| False negatives | 11 | 8 | -3 | Neural Network |
| False positives | 244 | 262 | 18 | Classic ML |
| True positives | 39 | 42 | 3 | Neural Network |
| True negatives | 703 | 685 | -18 | Classic ML |

## Análisis por métrica

- **Recall:** aumenta de 0.780000 a 0.840000 (+0.060000), una mejora relevante
  para un sistema de cribado.
- **Precision:** pasa de 0.137809 a 0.138158 (+0.000349); es un empate práctico.
- **F1:** mejora ligeramente de 0.234234 a 0.237288 (+0.003054).
- **ROC-AUC:** aumenta de 0.825723 a 0.835333 (+0.009609).
- **PR-AUC:** aumenta de 0.145957 a 0.163629 (+0.017672), especialmente
  informativo por el fuerte desbalanceo de clases.
- **Falsos negativos:** disminuyen de 11 a 8; la NN detecta tres casos positivos
  adicionales.
- **Falsos positivos:** aumentan de 244 a 262, es decir, 18 alertas adicionales.

## Overfitting y estabilidad

La Logistic Regression presenta gaps train-test reducidos: F1 0.010081,
ROC-AUC 0.008775 y PR-AUC 0.034719. Esto respalda una mayor estabilidad.

La NN presenta gaps train-validation de F1 0.010164 y PR-AUC 0.019981. Su
mejor época fue la 23 y ejecutó 33 épocas. Después del mínimo de validation
loss (0.464288), la pérdida de validation terminó en 0.466609, mientras la
pérdida de train terminó en 0.481016; además, validation PR-AUC pasó de
0.226771 en la mejor época a 0.218470. Esto muestra overfitting leve, limitado
mediante early stopping con restauración del mejor checkpoint.

## Trade-offs y conclusión

La **red neuronal tabular es la ganadora experimental** para el objetivo de
cribado: mejora recall y PR-AUC, reduce los falsos negativos y también supera
ligeramente al modelo clásico en F1 y ROC-AUC. Sin embargo, genera 18 falsos
positivos adicionales, la precision está prácticamente empatada, muestra algo
más de overfitting, es más compleja y sus probabilidades no están calibradas.

Esta conclusión no implica sustituir la Logistic Regression desplegada. El
modelo clásico conserva ventajas de estabilidad, interpretabilidad, calibración
y sencillez operativa. No existe una prueba de significancia estadística que
permita afirmar que las mejoras pequeñas de la NN sean concluyentes.

## Limitaciones

- La comparación contiene una única evaluación final sobre el split test.
- No se calcularon intervalos de confianza ni pruebas de significancia.
- El análisis clásico de estabilidad usa train-test, mientras que la NN aporta
  además curvas train-validation por época.
- Los modelos aplican estrategias y thresholds distintos, seleccionados sobre
  validation según su propio protocolo.
- Los resultados apoyan una comparación experimental, no una conclusión
  diagnóstica ni una decisión automática de despliegue.
