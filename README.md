# 🧠 Stroke Risk AI — Hospital F5

Sistema de cribado preventivo del riesgo de ictus desarrollado como proyecto **Data Scientist / AI Developer**.

El sistema utilizará Machine Learning para estimar el riesgo de ictus a partir de datos clínicos y demográficos, proporcionando una herramienta de apoyo para personal sanitario autorizado.

> ⚠️ Este proyecto es un prototipo educativo. Sus resultados no constituyen un diagnóstico médico.

## 🎯 Objetivo

Desarrollar una solución completa que incluya:

- Análisis exploratorio de datos (EDA).
- Preprocesamiento y validación de datos.
- Modelos de Machine Learning.
- Tratamiento del desbalanceo de clases.
- Optimización de hiperparámetros.
- Evaluación y explicabilidad del modelo.
- API con FastAPI.
- Aplicación web con React.
- Persistencia de pacientes y predicciones.
- CLI.
- Testing.
- MLflow.
- Docker.
- Deep Learning y CNN como evolución de nivel experto.

## 🛠️ Tecnologías

- Python
- Pandas
- NumPy
- Scikit-learn
- Optuna
- MLflow
- FastAPI
- React
- Docker
- Git / GitHub

## 🌿 Ramas principales

- `main` — versión estable.
- `dev` — integración del desarrollo.
- `feature/*` — desarrollo de funcionalidades.

## 📋 Metodología

El proyecto utiliza:

- Specification-Driven Development (SDD).
- Kanban mediante GitHub Projects.
- Desarrollo incremental.
- Validación y testing durante todo el ciclo de desarrollo.

## ⚙️ Configuración

La configuración de la aplicación se gestiona mediante variables de entorno para evitar dependencias del entorno local y facilitar su ejecución en diferentes plataformas.

Las variables disponibles están documentadas en `.env.example`:

- `APP_ENV` — entorno de ejecución de la aplicación.
- `DATABASE_URL` — conexión a la base de datos.
- `MODEL_PATH` — ruta relativa al modelo de Machine Learning.
- `MODEL_THRESHOLD` — umbral utilizado para la clasificación.
- `MLFLOW_TRACKING_URI` — dirección del servidor de tracking de MLflow.
- `RANDOM_SEED` — semilla global para garantizar reproducibilidad.

Para configuración local puede utilizarse un archivo `.env`, que está excluido del control de versiones.

## 📁 Estado

🚧 Proyecto en desarrollo.
