import os
from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./stroke_risk.db",
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/model.joblib",
)

MODEL_THRESHOLD = float(
    os.getenv("MODEL_THRESHOLD", "0.50")
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

RANDOM_SEED = int(
    os.getenv("RANDOM_SEED", "42")
)