FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini .
COPY artifacts/final_model/stroke_model_logreg_v1.joblib \
    ./artifacts/final_model/stroke_model_logreg_v1.joblib
COPY artifacts/final_model/threshold_logreg_v1.json \
    ./artifacts/final_model/threshold_logreg_v1.json
COPY artifacts/final_model/reference_values_logreg_v1.json \
    ./artifacts/final_model/reference_values_logreg_v1.json

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn src.api.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
