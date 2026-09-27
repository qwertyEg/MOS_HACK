# Демонстрация идёт на DGX Spark (aarch64), поэтому образ собирается под ту же
# архитектуру, на которой будет запускаться. Мультиарх не делаем — не нужен.
FROM python:3.12-slim

# libgl и libglib нужны opencv даже в headless-сборке
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY reference/ ./reference/
COPY tools/ ./tools/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
