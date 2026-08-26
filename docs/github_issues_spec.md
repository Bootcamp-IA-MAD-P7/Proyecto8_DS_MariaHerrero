# GitHub Issues Specification — Hospital F5 Stroke Risk AI

Repositorio: `Bootcamp-IA-MAD-P7/Proyecto8_DS_MariaHerrero`

Proyecto GitHub: `Stroke Risk AI - Project 8`

## Reglas de creación

- Crear una Issue por cada User Story definida en este documento.
- Respetar exactamente títulos, milestones y labels.
- No duplicar Issues existentes.
- Añadir todas las Issues al GitHub Project `Stroke Risk AI - Project 8`.
- Estado inicial en el Project: `Backlog`.
- No asignar responsable.
- Mantener las dependencias indicadas en el cuerpo de cada Issue.
- Si falta una label, milestone o el Project no es accesible, detener la creación y reportarlo antes de continuar.

---

# EPIC 01 — Foundation

## US-01 — Inicializar proyecto

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `documentation`  
**Prioridad:** Must  
**Orden:** 1  
**Dependencias:** Ninguna

### Descripción
Como developer quiero disponer de un proyecto estructurado y versionado para desarrollar todos los componentes de manera independiente y reproducible.

### Tareas técnicas
- [ ] Crear estructura de carpetas.
- [ ] Configurar `.gitignore`.
- [ ] Crear `.env.example`.
- [ ] Crear README inicial.
- [ ] Definir versión de Python.
- [ ] Crear entorno virtual.
- [ ] Crear dependencias iniciales.
- [ ] Definir seed global.
- [ ] Confirmar ramas `main` y `dev`.

### Criterios de aceptación
- El repositorio puede clonarse y ejecutarse en local.
- No existen secretos en Git.
- La estructura respeta las specs maestras.
- `main` y `dev` existen en remoto.

---

## US-02 — Preparar configuración desacoplada

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `documentation`  
**Prioridad:** Must  
**Orden:** 2  
**Dependencias:** US-01

### Descripción
Como developer quiero centralizar la configuración del proyecto para evitar dependencias del entorno local y preparar un despliegue agnóstico al proveedor.

### Tareas técnicas
- [ ] Crear configuración central.
- [ ] Preparar variables de entorno.
- [ ] Definir `DATABASE_URL`.
- [ ] Definir configuración del modelo.
- [ ] Preparar `MLFLOW_TRACKING_URI`.
- [ ] Evitar rutas absolutas locales.
- [ ] Documentar variables necesarias.

### Criterios de aceptación
- El proyecto no depende de rutas tipo `C:\...`.
- La configuración sensible se obtiene mediante variables de entorno.
- Existe `.env.example` actualizado.

---

# EPIC 02 — Data & EDA

## US-03 — Auditar dataset

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `data`, `eda`  
**Prioridad:** Must  
**Orden:** 3  
**Dependencias:** US-01

### Descripción
Como Data Scientist quiero conocer la calidad y estructura del dataset antes de transformarlo para poder tomar decisiones de preprocessing justificadas.

### Tareas técnicas
- [ ] Analizar dimensiones.
- [ ] Revisar columnas.
- [ ] Revisar tipos.
- [ ] Detectar nulos.
- [ ] Detectar duplicados.
- [ ] Analizar categorías.
- [ ] Analizar rangos.
- [ ] Detectar valores anómalos.
- [ ] Analizar distribución de `stroke`.

### Criterios de aceptación
- Existe un análisis reproducible del dataset.
- Se documentan problemas de calidad detectados.
- Se identifica claramente el nivel de desbalanceo de `stroke`.

---

## US-04 — Realizar EDA

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `data`, `eda`  
**Prioridad:** Must  
**Orden:** 4  
**Dependencias:** US-03

### Descripción
Como Data Scientist quiero analizar las variables y su relación con `stroke` para comprender los patrones presentes en los datos.

### Tareas técnicas
- [ ] Estadística descriptiva.
- [ ] Distribuciones numéricas.
- [ ] Distribuciones categóricas.
- [ ] Analizar `gender`.
- [ ] Analizar `age`.
- [ ] Analizar `hypertension`.
- [ ] Analizar `heart_disease`.
- [ ] Analizar `ever_married`.
- [ ] Analizar `work_type`.
- [ ] Analizar `Residence_type`.
- [ ] Analizar `avg_glucose_level`.
- [ ] Analizar `bmi`.
- [ ] Analizar `smoking_status`.
- [ ] Analizar relaciones con `stroke`.
- [ ] Analizar correlaciones.
- [ ] Crear visualizaciones.
- [ ] Documentar conclusiones.

### Criterios de aceptación
- Cada gráfico responde a una pregunta concreta.
- El EDA incluye estadísticas, visualizaciones y conclusiones.
- El desbalanceo del target queda documentado.

---

# EPIC 03 — Data Pipeline

## US-05 — Crear Data Validation Harness

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `data`, `testing`  
**Prioridad:** Must  
**Orden:** 5  
**Dependencias:** US-03

### Descripción
Como equipo ML quiero validar automáticamente el dataset para detectar incompatibilidades antes del entrenamiento.

### Tareas técnicas
- [ ] Crear schema esperado.
- [ ] Validar columnas.
- [ ] Validar tipos.
- [ ] Validar target.
- [ ] Validar nulos.
- [ ] Validar duplicados.
- [ ] Validar categorías.
- [ ] Validar rangos.
- [ ] Validar distribución.
- [ ] Crear tests.

### Criterios de aceptación
- Un dataset incompatible produce un error identificable.
- Las reglas de validación están centralizadas.
- Los tests del harness pasan.

---

## US-06 — Crear split reproducible

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `data`, `machine-learning`  
**Prioridad:** Must  
**Orden:** 6  
**Dependencias:** US-05

### Descripción
Como Data Scientist quiero separar correctamente train, validation y test para evitar contaminación entre entrenamiento y evaluación.

### Tareas técnicas
- [ ] Separar X/y.
- [ ] Excluir `stroke` de features.
- [ ] Crear train.
- [ ] Crear validation.
- [ ] Crear test.
- [ ] Utilizar stratification.
- [ ] Utilizar seed fija.
- [ ] Verificar distribución de clases.

### Criterios de aceptación
- Train, validation y test mantienen proporciones de clase razonables.
- El split es reproducible.
- Test queda bloqueado para selección, tuning, balanceo y threshold.

---

## US-07 — Crear preprocessing pipeline

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `data`, `machine-learning`  
**Prioridad:** Must  
**Orden:** 7  
**Dependencias:** US-06

### Descripción
Como Data Scientist quiero un preprocessing reutilizable para garantizar que entrenamiento e inferencia aplican exactamente las mismas transformaciones.

### Tareas técnicas
- [ ] Definir estrategia de missing values.
- [ ] Codificar variables categóricas.
- [ ] Procesar variables numéricas.
- [ ] Escalar cuando proceda.
- [ ] Implementar `ColumnTransformer`.
- [ ] Implementar `Pipeline`.
- [ ] Crear tests del preprocessing.

### Criterios de aceptación
- El pipeline hace `fit` exclusivamente con training.
- El mismo preprocessing puede reutilizarse en prediction.
- Los tests pasan.

---

## US-08 — Implementar Leakage Harness

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `data`, `testing`, `safety`  
**Prioridad:** Must  
**Orden:** 8  
**Dependencias:** US-06, US-07

### Descripción
Como Data Scientist quiero comprobar automáticamente la ausencia de data leakage para garantizar una evaluación válida del modelo.

### Tareas técnicas
- [ ] Comprobar que `stroke` no entra en X.
- [ ] Comprobar que preprocessing solo hace fit con train.
- [ ] Comprobar que balanceo solo afecta a train.
- [ ] Comprobar que validation no contamina training.
- [ ] Comprobar que test permanece aislado.

### Criterios de aceptación
- Existen tests automáticos para los casos de leakage definidos.
- Todos los tests pasan antes de entrenar modelos candidatos.

---

# EPIC 04 — Machine Learning

## US-09 — Crear baseline

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `machine-learning`  
**Prioridad:** Must  
**Orden:** 9  
**Dependencias:** US-07, US-08

### Descripción
Como Data Scientist quiero disponer de un baseline para medir si los modelos posteriores aportan una mejora real.

### Tareas técnicas
- [ ] Entrenar `DummyClassifier`.
- [ ] Entrenar Logistic Regression.
- [ ] Calcular Recall.
- [ ] Calcular Precision.
- [ ] Calcular F1.
- [ ] Calcular ROC-AUC.
- [ ] Calcular PR-AUC.
- [ ] Crear matriz de confusión.
- [ ] Registrar falsos negativos.
- [ ] Registrar experimentos en MLflow si ya está disponible.

### Criterios de aceptación
- Existen métricas comparables para ambos baselines.
- Accuracy no se utiliza como única métrica de selección.
- Los falsos negativos quedan registrados.

---

## US-10 — Comparar modelos clásicos

**Milestone:** 🟡 Nivel Medio  
**Labels:** `machine-learning`  
**Prioridad:** Must  
**Orden:** 10  
**Dependencias:** US-09

### Descripción
Como Data Scientist quiero comparar varias familias de modelos para seleccionar candidatos robustos.

### Tareas técnicas
- [ ] Evaluar Logistic Regression.
- [ ] Evaluar Decision Tree.
- [ ] Evaluar Random Forest.
- [ ] Evaluar Gradient Boosting o XGBoost.
- [ ] Evaluar SVM si resulta razonable.
- [ ] Crear tabla comparativa.
- [ ] Analizar overfitting.

### Criterios de aceptación
- Todos los modelos se evalúan con el mismo protocolo.
- La tabla incluye Recall, Precision, F1, ROC-AUC, PR-AUC y FN.
- Se identifican candidatos para optimización.

---

## US-11 — Implementar validación cruzada

**Milestone:** 🟡 Nivel Medio  
**Labels:** `machine-learning`  
**Prioridad:** Must  
**Orden:** 11  
**Dependencias:** US-10

### Descripción
Como Data Scientist quiero aplicar validación cruzada estratificada para medir la estabilidad de los modelos.

### Tareas técnicas
- [ ] Implementar `StratifiedKFold`.
- [ ] Evaluar modelos candidatos.
- [ ] Calcular media de métricas.
- [ ] Calcular desviación.
- [ ] Analizar estabilidad.

### Criterios de aceptación
- Los candidatos tienen resultados de CV reproducibles.
- Se documentan media y dispersión de las métricas principales.

---

# EPIC 05 — Optimización

## US-12 — Tratar desbalanceo

**Milestone:** 🟡 Nivel Medio  
**Labels:** `machine-learning`, `imbalance`  
**Prioridad:** Must  
**Orden:** 12  
**Dependencias:** US-11

### Descripción
Como Data Scientist quiero comparar técnicas de tratamiento del desbalanceo para mejorar la detección de la clase positiva sin introducir leakage.

### Tareas técnicas
- [ ] Evaluar `class_weight`.
- [ ] Evaluar Random Oversampling.
- [ ] Evaluar SMOTE.
- [ ] Comparar Recall y falsos negativos.
- [ ] Documentar estrategia ganadora.

### Criterios de aceptación
- El resampling solo se aplica sobre training.
- Se comparan estrategias bajo el mismo protocolo.
- La decisión se justifica por métricas, especialmente Recall y FN.

---

## US-13 — Optimizar hiperparámetros

**Milestone:** 🟡 Nivel Medio  
**Labels:** `machine-learning`  
**Prioridad:** Must  
**Orden:** 13  
**Dependencias:** US-12

### Descripción
Como Data Scientist quiero optimizar los hiperparámetros de los modelos candidatos para mejorar su rendimiento de manera reproducible.

### Tareas técnicas
- [ ] Seleccionar modelos candidatos.
- [ ] Definir espacios de búsqueda.
- [ ] Configurar Optuna.
- [ ] Ejecutar trials.
- [ ] Registrar resultados.
- [ ] Seleccionar configuraciones candidatas.

### Criterios de aceptación
- Test no participa en la optimización.
- Los trials son reproducibles.
- Las mejores configuraciones quedan documentadas.

---

## US-14 — Optimizar threshold

**Milestone:** 🟡 Nivel Medio  
**Labels:** `machine-learning`, `safety`  
**Prioridad:** Must  
**Orden:** 14  
**Dependencias:** US-13

### Descripción
Como Data Scientist quiero ajustar el umbral de decisión para controlar mejor el equilibrio entre falsos negativos y falsos positivos.

### Tareas técnicas
- [ ] Generar Precision-Recall Curve.
- [ ] Analizar distintos thresholds.
- [ ] Comparar FN/FP.
- [ ] Seleccionar threshold con validation.
- [ ] Documentar criterio.
- [ ] Versionar threshold.

### Criterios de aceptación
- Test no se utiliza para elegir threshold.
- El threshold seleccionado tiene justificación explícita.
- El valor queda asociado a la versión del modelo.

---

## US-15 — Calibrar probabilidades

**Milestone:** 🟡 Nivel Medio  
**Labels:** `machine-learning`  
**Prioridad:** Should  
**Orden:** 15  
**Dependencias:** US-13

### Descripción
Como Data Scientist quiero comprobar la calibración de las probabilidades para evitar mostrar scores engañosos en la aplicación.

### Tareas técnicas
- [ ] Crear calibration curve.
- [ ] Calcular Brier Score.
- [ ] Comparar calibrado/no calibrado.
- [ ] Implementar calibración si procede.
- [ ] Documentar interpretación del score.

### Criterios de aceptación
- Se evalúa explícitamente la calibración.
- Si se muestra una probabilidad al usuario, su interpretación está documentada.

---

# EPIC 06 — Modelo Final & Explainability

## US-16 — Seleccionar modelo final

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `machine-learning`  
**Prioridad:** Must  
**Orden:** 16  
**Dependencias:** US-13, US-14, US-15

### Descripción
Como Data Scientist quiero seleccionar y evaluar el modelo final para disponer de un modelo estable y justificable para producción.

### Tareas técnicas
- [ ] Crear ranking final.
- [ ] Justificar selección.
- [ ] Desbloquear test.
- [ ] Ejecutar evaluación final.
- [ ] Calcular métricas de test.
- [ ] Comparar train/test.
- [ ] Comprobar gap inferior a 5 puntos porcentuales.
- [ ] Generar matriz de confusión.
- [ ] Guardar modelo.
- [ ] Guardar preprocessing.
- [ ] Guardar threshold.
- [ ] Documentar elección.

### Criterios de aceptación
- Test se utiliza solo en la evaluación final.
- El gap train/test cumple el requisito o queda documentada la desviación.
- Modelo, preprocessing y threshold quedan versionados.

---

## US-17 — Implementar explicabilidad

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `machine-learning`, `explainability`, `safety`  
**Prioridad:** Must  
**Orden:** 17  
**Dependencias:** US-16

### Descripción
Como profesional sanitario quiero conocer qué variables influyen en la estimación para interpretar el resultado como apoyo a la criba.

### Tareas técnicas
- [ ] Implementar explicabilidad global.
- [ ] Calcular feature importance cuando proceda.
- [ ] Implementar SHAP u otro método compatible.
- [ ] Crear explicación individual.
- [ ] Identificar factores que aumentan/disminuyen el score.
- [ ] Preparar formato consumible por API.
- [ ] Evitar lenguaje causal.

### Criterios de aceptación
- Existe explicación global e individual.
- La salida describe influencia sobre el modelo, nunca causalidad médica.
- La explicación puede ser consumida por FastAPI.

---

# EPIC 07 — Productivización

## US-18 — Crear FastAPI

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `backend`, `api`  
**Prioridad:** Must  
**Orden:** 18  
**Dependencias:** US-16, US-17

### Descripción
Como aplicación cliente quiero consumir el modelo mediante una API estable para desacoplar frontend, CLI y lógica ML.

### Tareas técnicas
- [ ] Crear FastAPI.
- [ ] Crear schemas Pydantic.
- [ ] Crear `ModelService`.
- [ ] Crear `PredictionService`.
- [ ] Implementar validación.
- [ ] Implementar gestión de errores.
- [ ] Crear `GET /api/v1/health`.
- [ ] Crear `POST /api/v1/predictions`.
- [ ] Documentar OpenAPI.

### Criterios de aceptación
- La API carga el modelo sin lógica ML en el frontend.
- Una entrada válida devuelve una predicción válida.
- Los errores de entrada devuelven respuestas controladas.

---

## US-19 — Implementar base de datos

**Milestone:** 🟠 Nivel Avanzado  
**Labels:** `backend`, `database`  
**Prioridad:** Must  
**Orden:** 19  
**Dependencias:** US-18

### Descripción
Como sistema quiero persistir pacientes, evaluaciones y predicciones para mantener historial y trazabilidad.

### Tareas técnicas
- [ ] Diseñar esquema relacional.
- [ ] Crear entidades `Patient`, `Assessment`, `Prediction`, `ModelVersion`.
- [ ] Configurar base de datos.
- [ ] Crear models/repositories.
- [ ] Crear migrations.
- [ ] Implementar persistencia.
- [ ] Crear tests.

### Criterios de aceptación
- Una evaluación no sobrescribe evaluaciones anteriores.
- Cada predicción puede asociarse al paciente y al modelo que la produjo.
- Los tests de persistencia pasan.

---

## US-20 — Garantizar trazabilidad de predicciones

**Milestone:** 🟠 Nivel Avanzado  
**Labels:** `backend`, `database`, `mlops`  
**Prioridad:** Must  
**Orden:** 20  
**Dependencias:** US-19

### Descripción
Como sistema quiero conservar información suficiente para reconstruir el contexto de cada predicción histórica.

### Tareas técnicas
- [ ] Guardar timestamp.
- [ ] Guardar datos de assessment.
- [ ] Guardar score.
- [ ] Guardar clasificación.
- [ ] Guardar versión del modelo.
- [ ] Guardar threshold.
- [ ] Guardar origen `professional` o `self_reported`.

### Criterios de aceptación
- Cada predicción histórica identifica modelo, threshold, fecha, origen y datos usados.
- El sistema diferencia evaluaciones profesionales y autodeclaradas.

---

## US-21 — Crear CLI

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `cli`, `backend`, `safety`  
**Prioridad:** Must  
**Orden:** 21  
**Dependencias:** US-18

### Descripción
Como profesional sanitario quiero introducir datos por línea de comandos para cumplir el flujo solicitado en el briefing.

### Tareas técnicas
- [ ] Crear CLI.
- [ ] Solicitar variables necesarias.
- [ ] Reutilizar schemas.
- [ ] Reutilizar PredictionService.
- [ ] Validar entradas.
- [ ] Mostrar resultado.
- [ ] Mostrar disclaimer.

### Criterios de aceptación
- La CLI acepta entradas válidas.
- Rechaza entradas inválidas.
- Devuelve una predicción consistente con la API.
- No presenta el resultado como diagnóstico.

---

## US-22 — Crear CLI Test Harness

**Milestone:** 🟡 Nivel Medio  
**Labels:** `cli`, `testing`  
**Prioridad:** Must  
**Orden:** 22  
**Dependencias:** US-21

### Descripción
Como developer quiero probar automáticamente la CLI para asegurar entradas y salidas consistentes.

### Tareas técnicas
- [ ] Test paciente válido.
- [ ] Test campo obligatorio ausente.
- [ ] Test edad inválida.
- [ ] Test categoría inválida.
- [ ] Test BMI inválido.
- [ ] Test modelo no disponible.
- [ ] Test respuesta consistente.

### Criterios de aceptación
- Todos los casos definidos están automatizados.
- La CLI falla de forma controlada ante inputs inválidos.

---

# EPIC 08 — Aplicación React

## US-23 — Crear estructura visual React

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `frontend`  
**Prioridad:** Must  
**Orden:** 23  
**Dependencias:** US-18

### Descripción
Como profesional sanitario quiero una aplicación web clara para navegar entre nueva evaluación, resultado e historial.

### Tareas técnicas
- [ ] Crear proyecto React.
- [ ] Configurar routing.
- [ ] Crear layout.
- [ ] Crear navegación.
- [ ] Preparar diseño responsive.
- [ ] Configurar cliente API.

### Criterios de aceptación
- La aplicación arranca correctamente.
- Existe navegación entre las vistas principales.
- El frontend no contiene lógica de inferencia ML.

---

## US-24 — Crear formulario de evaluación

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `frontend`, `api`, `safety`  
**Prioridad:** Must  
**Orden:** 24  
**Dependencias:** US-23, US-18

### Descripción
Como enfermera quiero introducir los datos del paciente para solicitar una criba preventiva de riesgo de ictus.

### Tareas técnicas
- [ ] Crear campos del formulario.
- [ ] Validar datos en UX.
- [ ] Integrar con FastAPI.
- [ ] Crear estados loading/error/success.
- [ ] Mostrar mensajes de validación.

### Criterios de aceptación
- El formulario incluye todas las features requeridas por el modelo.
- Una entrada válida puede enviarse a la API.
- Los errores de entrada son visibles y comprensibles.

---

## US-25 — Mostrar resultado de riesgo

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `frontend`, `explainability`, `safety`  
**Prioridad:** Must  
**Orden:** 25  
**Dependencias:** US-24, US-17

### Descripción
Como enfermera quiero recibir un resultado comprensible para utilizarlo como apoyo a una criba preventiva.

### Tareas técnicas
- [ ] Crear componente de resultado.
- [ ] Mostrar nivel de riesgo.
- [ ] Mostrar score del modelo.
- [ ] Mostrar principales factores influyentes.
- [ ] Mostrar disclaimer clínico.
- [ ] Evitar mensajes diagnósticos.

### Criterios de aceptación
- La vista muestra riesgo, score y explicación.
- El mensaje deja claro que no constituye diagnóstico.
- No se presenta correlación como causalidad.

---

## US-26 — Crear historial de evaluaciones

**Milestone:** 🟠 Nivel Avanzado  
**Labels:** `frontend`, `database`, `api`  
**Prioridad:** Should  
**Orden:** 26  
**Dependencias:** US-19, US-20, US-23

### Descripción
Como profesional sanitario quiero consultar evaluaciones anteriores para disponer del historial de cribas realizadas.

### Tareas técnicas
- [ ] Consultar evaluaciones mediante API.
- [ ] Crear listado cronológico.
- [ ] Crear vista de detalle.
- [ ] Mostrar modelo utilizado.
- [ ] Mostrar score histórico.

### Criterios de aceptación
- El historial se obtiene desde la base de datos.
- Se puede consultar el detalle de una evaluación anterior.

---

# EPIC 09 — Quality, Safety & MLOps

## US-27 — Implementar Clinical Safety Guardrails

**Milestone:** 🟢 Nivel Esencial  
**Labels:** `safety`, `testing`  
**Prioridad:** Must  
**Orden:** 27  
**Dependencias:** US-18, US-25

### Descripción
Como sistema de apoyo clínico quiero aplicar guardrails para impedir que la aplicación se presente como herramienta diagnóstica.

### Tareas técnicas
- [ ] Centralizar mensajes clínicos.
- [ ] Validar rangos.
- [ ] Implementar disclaimer.
- [ ] Evitar recomendaciones de medicación.
- [ ] Evitar afirmaciones diagnósticas.
- [ ] Crear tests de guardrails.

### Criterios de aceptación
- El sistema nunca afirma que una persona sufrirá o no sufrirá un ictus.
- Los valores imposibles no se aceptan silenciosamente.
- Los mensajes de seguridad están cubiertos por tests.

---

## US-28 — Completar Test Suite

**Milestone:** 🟡 Nivel Medio  
**Labels:** `testing`  
**Prioridad:** Must  
**Orden:** 28  
**Dependencias:** US-22, US-27

### Descripción
Como developer quiero una suite de tests suficiente para detectar regresiones en los componentes críticos.

### Tareas técnicas
- [ ] Unit tests.
- [ ] Preprocessing tests.
- [ ] ML tests.
- [ ] Validation tests.
- [ ] API tests.
- [ ] Database tests.
- [ ] CLI tests.
- [ ] Integration tests.

### Criterios de aceptación
- Los componentes críticos tienen cobertura de tests.
- La suite completa puede ejecutarse con un único comando.
- No existen fallos conocidos en tests antes de integrar a `dev`.

---

## US-29 — Integrar MLflow

**Milestone:** 🟠 Nivel Avanzado  
**Labels:** `mlops`, `machine-learning`  
**Prioridad:** Must  
**Orden:** 29  
**Dependencias:** US-09

### Descripción
Como Data Scientist quiero registrar experimentos en MLflow para comparar y reproducir modelos.

### Tareas técnicas
- [ ] Configurar MLflow.
- [ ] Crear experiment.
- [ ] Registrar algoritmos.
- [ ] Registrar hiperparámetros.
- [ ] Registrar preprocessing.
- [ ] Registrar estrategia de balanceo.
- [ ] Registrar seed.
- [ ] Registrar métricas.
- [ ] Registrar artifacts.
- [ ] Registrar modelos.

### Criterios de aceptación
- Es posible identificar qué experimento produjo cada modelo registrado.
- Los experimentos relevantes del proyecto quedan registrados.
- El tracking no depende de rutas absolutas.

---

## US-30 — Dockerizar aplicación

**Milestone:** 🟠 Nivel Avanzado  
**Labels:** `docker`, `backend`, `frontend`, `database`, `mlops`  
**Prioridad:** Must  
**Orden:** 30  
**Dependencias:** US-19, US-23, US-29

### Descripción
Como developer quiero ejecutar la aplicación completa mediante contenedores para garantizar portabilidad.

### Tareas técnicas
- [ ] Dockerizar frontend.
- [ ] Dockerizar backend.
- [ ] Configurar database.
- [ ] Configurar MLflow.
- [ ] Crear `docker-compose.yml`.
- [ ] Verificar comunicación entre servicios.
- [ ] Probar arranque completo.

### Criterios de aceptación
- La aplicación completa puede arrancarse mediante Docker Compose.
- Los servicios se comunican sin depender de configuraciones locales específicas.

---

## US-31 — Configurar CI

**Milestone:** 🟠 Nivel Avanzado  
**Labels:** `testing`, `documentation`  
**Prioridad:** Should  
**Orden:** 31  
**Dependencias:** US-28

### Descripción
Como developer quiero ejecutar comprobaciones automáticas en GitHub para reducir regresiones.

### Tareas técnicas
- [ ] Crear workflow de GitHub Actions.
- [ ] Ejecutar backend tests.
- [ ] Ejecutar ML tests.
- [ ] Ejecutar frontend checks.
- [ ] Fallar pipeline ante errores críticos.

### Criterios de aceptación
- Pull Requests ejecutan automáticamente las comprobaciones definidas.
- Un fallo crítico provoca estado fallido del workflow.

---

## US-32 — Desplegar aplicación

**Milestone:** 🟠 Nivel Avanzado  
**Labels:** `docker`, `backend`, `frontend`, `database`  
**Prioridad:** Must  
**Orden:** 32  
**Dependencias:** US-30, US-31

### Descripción
Como usuario quiero acceder públicamente al prototipo para poder demostrar su funcionamiento sin depender del entorno local.

### Tareas técnicas
- [ ] Seleccionar proveedor.
- [ ] Configurar variables de producción.
- [ ] Desplegar base de datos.
- [ ] Desplegar FastAPI.
- [ ] Desplegar React.
- [ ] Configurar modelo.
- [ ] Ejecutar smoke tests.
- [ ] Verificar acceso público.

### Criterios de aceptación
- Existe una URL pública funcional.
- El flujo principal de predicción funciona en producción.
- No existen secretos expuestos en el repositorio.

---

# EPIC 10 — Nivel Experto

## US-33 — Crear red neuronal tabular

**Milestone:** 🔴 Nivel Experto  
**Labels:** `deep-learning`, `machine-learning`, `mlops`  
**Prioridad:** Must  
**Orden:** 33  
**Dependencias:** US-16, US-29

### Descripción
Como Data Scientist quiero entrenar una red neuronal sobre los datos tabulares para comparar Deep Learning con los modelos clásicos.

### Tareas técnicas
- [ ] Diseñar arquitectura NN.
- [ ] Reutilizar splits existentes.
- [ ] Entrenar.
- [ ] Aplicar regularización.
- [ ] Controlar overfitting.
- [ ] Optimizar hiperparámetros.
- [ ] Evaluar con las mismas métricas.
- [ ] Registrar en MLflow.

### Criterios de aceptación
- La NN utiliza el mismo protocolo de evaluación que los modelos clásicos.
- Se registran métricas, parámetros y modelo.
- El overfitting se analiza explícitamente.

---

## US-34 — Comparar red neuronal vs ML clásico

**Milestone:** 🔴 Nivel Experto  
**Labels:** `deep-learning`, `machine-learning`, `documentation`  
**Prioridad:** Must  
**Orden:** 34  
**Dependencias:** US-33

### Descripción
Como Data Scientist quiero comparar de forma objetiva la red neuronal con el mejor modelo clásico para justificar cuál es más adecuado.

### Tareas técnicas
- [ ] Comparar Recall.
- [ ] Comparar Precision.
- [ ] Comparar F1.
- [ ] Comparar ROC-AUC.
- [ ] Comparar PR-AUC.
- [ ] Comparar falsos negativos.
- [ ] Comparar overfitting.
- [ ] Documentar ganador.

### Criterios de aceptación
- La comparación utiliza las mismas métricas y splits.
- La conclusión se basa en resultados, aunque gane el modelo clásico.

---

## US-35 — Crear prototipo CNN para imágenes CT

**Milestone:** 🔴 Nivel Experto  
**Labels:** `deep-learning`, `mlops`, `data`  
**Prioridad:** Must  
**Orden:** 35  
**Dependencias:** US-29

### Descripción
Como Data Scientist quiero crear un prototipo CNN con imágenes CT para explorar una segunda modalidad de detección relacionada con ictus.

### Tareas técnicas
- [ ] Auditar dataset CT.
- [ ] Preparar imágenes.
- [ ] Crear splits.
- [ ] Diseñar CNN.
- [ ] Entrenar.
- [ ] Aplicar regularización.
- [ ] Evaluar.
- [ ] Registrar experimento en MLflow.
- [ ] Documentar resultados.

### Criterios de aceptación
- El pipeline de imágenes está separado del tabular.
- La CNN se evalúa con un protocolo documentado.
- El experimento queda registrado en MLflow.

---

## US-36 — Preparar arquitectura multimodal

**Milestone:** 🔴 Nivel Experto  
**Labels:** `deep-learning`, `backend`, `api`, `documentation`  
**Prioridad:** Should  
**Orden:** 36  
**Dependencias:** US-18, US-33, US-35

### Descripción
Como developer quiero que la arquitectura pueda incorporar un predictor basado en CT sin reconstruir el sistema existente.

### Tareas técnicas
- [ ] Crear interfaz común de predictors.
- [ ] Mantener predictor tabular desacoplado.
- [ ] Mantener CNN desacoplada.
- [ ] Documentar futura integración multimodal.
- [ ] Verificar que FastAPI puede incorporar un nuevo predictor sin romper el existente.

### Criterios de aceptación
- Añadir el predictor CNN no obliga a reconstruir React, pacientes o base de datos.
- La arquitectura futura queda documentada.
- Los modelos tabular y de imagen permanecen independientes.

---

# Definition of Done global

Una Issue solo se considera terminada cuando:

- [ ] Cumple sus criterios de aceptación.
- [ ] El código está terminado.
- [ ] Los tests correspondientes pasan.
- [ ] No rompe funcionalidades existentes.
- [ ] Los cambios están subidos a Git.
- [ ] Los commits son descriptivos.
- [ ] La funcionalidad está integrada en `dev`.
- [ ] La documentación se actualiza cuando corresponde.
- [ ] La Issue se mueve a `Done` en GitHub Project.

# Flujo Kanban

`Backlog → Ready → In progress → Testing → Done`

# Regla SDD

Antes de implementar una Issue:

1. Leer las specs maestras.
2. Leer la User Story.
3. Revisar criterios de aceptación.
4. Revisar dependencias.
5. Crear rama de trabajo.
6. Implementar.
7. Ejecutar tests.
8. Validar criterios de aceptación.
9. Integrar en `dev`.
10. Cerrar Issue.

Si una implementación requiere cambiar una decisión arquitectónica, no modificarla silenciosamente: documentar el conflicto, actualizar la SPEC y después continuar.
