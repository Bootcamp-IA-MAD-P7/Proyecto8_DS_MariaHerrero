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

## 📊 MLflow y seguimiento de experimentos

MLflow se utiliza para comparar y reproducir los experimentos de Machine Learning del proyecto. Cada ejecución registra sus parámetros, métricas, tags y artifacts relevantes en uno de estos experimentos:

- `stroke-risk-model-selection`: baseline, comparación de modelos clásicos, validación cruzada, estrategias de balanceo y optimización de hiperparámetros con Optuna.
- `stroke-risk-calibration-threshold`: comparación de métodos de calibración, exploración del threshold y evaluación del threshold calibrado.
- `stroke-risk-final-model`: entrenamiento y evaluación del modelo final seleccionado.

La procedencia del modelo final registrado queda vinculada de forma explícita:

```text
stroke-risk-final-model
└── final-model-logreg-v1
    └── stroke-risk-screening-model (versión 1)
```

El tracking utiliza por defecto el backend local `sqlite:///mlflow.db`. Puede configurarse otro backend mediante la variable de entorno `MLFLOW_TRACKING_URI`, sin depender de rutas absolutas.

Para consultar los experimentos, desde la raíz del repositorio:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

La interfaz queda disponible en [http://127.0.0.1:5000](http://127.0.0.1:5000).

Los experimentos pueden poblarse ejecutando los siguientes módulos:

```powershell
python -m src.models.baseline
python -m src.models.compare_classic_models
python -m src.models.cross_validation
python -m src.models.imbalance_comparison
python -m src.models.hyperparameter_tuning
python -m src.models.probability_calibration
python -m src.models.threshold_optimization
python -m src.models.calibrated_threshold
python -m src.models.final_model
```

> ⚠️ Al volver a ejecutar `src.models.final_model`, MLflow puede crear una nueva versión de `stroke-risk-screening-model` en el Model Registry.

Los datos locales de ejecución de MLflow (`mlflow.db`, `mlruns/` y `mlartifacts/`) se excluyen intencionadamente de Git. La aplicación desplegada continúa cargando los artifacts versionados joblib/JSON existentes y no necesita que el servidor o la interfaz de MLflow estén activos.

## 🐳 Docker

La aplicación completa requiere Docker con Docker Compose y se inicia desde la raíz del repositorio:

```powershell
docker compose up --build -d
```

Servicios disponibles:

- Frontend: [http://localhost:5173](http://localhost:5173)
- Backend API: [http://localhost:8000](http://localhost:8000)
- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- MLflow: [http://localhost:5001](http://localhost:5001)

Para consultar el estado y los logs:

```powershell
docker compose ps
docker compose logs
```

Para detener el stack:

```powershell
docker compose down
```

SQLite persiste sus datos en el volumen `app_data`, y las migraciones Alembic se aplican automáticamente al arrancar el backend. El comando `docker compose down -v` elimina los volúmenes y, por tanto, los datos persistidos.

La inferencia carga los artifacts versionados y no depende de que MLflow esté disponible:

- `stroke_model_logreg_v1.joblib`
- `threshold_logreg_v1.json`
- `reference_values_logreg_v1.json`

La imagen backend contiene únicamente estos artifacts de inferencia, no los datasets de entrenamiento. `reference_values_logreg_v1.json` conserva los valores agregados y reproducibles que utiliza la explicación de predicciones, evitando incluir `train.csv` en la imagen.

## 🚂 Despliegue en Railway

El despliegue mínimo utiliza dos servicios Railway dentro del mismo proyecto y entorno. El frontend React se sirve con Nginx desde un dominio público y reenvía `/api/v1` al backend FastAPI mediante la red privada de Railway. El backend mantiene SQLite en un volumen persistente; MLflow no se despliega porque no es necesario para la inferencia.

Configuración del servicio `backend`:

- Dockerfile: `Dockerfile` en la raíz del repositorio.
- `DATABASE_URL=sqlite:////app/data/stroke_app.db`.
- Railway Volume montado en `/app/data`.
- `PORT` es proporcionado por Railway; localmente se utiliza `8000` por defecto.
- Healthcheck: `/api/v1/health`.
- Una sola réplica, necesaria para utilizar SQLite de forma segura.

Configuración del servicio `frontend`:

- Root Directory: `/frontend`; Dockerfile: `Dockerfile`.
- `VITE_API_URL=/api/v1`.
- `BACKEND_HOST=${{backend.RAILWAY_PRIVATE_DOMAIN}}`.
- `BACKEND_PORT=${{backend.PORT}}`.
- Nginx escucha en el `PORT` proporcionado por Railway y utiliza por defecto el puerto `80` fuera de Railway.

Las referencias entre servicios evitan versionar hostnames concretos. Solo el frontend necesita dominio público; el navegador nunca accede directamente a la red privada. Para verificar el despliegue, debe responder correctamente `https://<dominio-frontend>/api/v1/health`. Después se realiza una evaluación desde `/assessment` y se comprueba que aparece en `/history`, incluida tras reiniciar o redesplegar el backend sin eliminar el volumen.

## ✅ Testing

La suite completa se ejecuta desde la raíz del proyecto con un único comando:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

La ejecución debe finalizar sin fallos. Los tests utilizan recursos temporales y aislados cuando corresponde, por lo que no deben escribir en la base de datos local de la aplicación.

La suite cubre las principales áreas críticas: validación de datos, preprocesamiento, Machine Learning, servicios de modelo y predicción, API, base de datos, CLI, seguridad clínica, explicabilidad e integración entre capas.

## 📁 Estado

🚧 Proyecto en desarrollo.
