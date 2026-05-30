# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim — лёгкий образ без лишних пакетов
FROM python:3.11-slim

# ── Метаданные образа ─────────────────────────────────────────────────────────
LABEL maintainer="student"
LABEL description="Heart Disease ML Pipeline — итоговый проект MLOps"
LABEL version="1.0"

# ── Переменные окружения ──────────────────────────────────────────────────────
# Отключаем буферизацию stdout (логи видны сразу)
ENV PYTHONUNBUFFERED=1
# Не создаём .pyc файлы внутри контейнера
ENV PYTHONDONTWRITEBYTECODE=1
# Пути внутри контейнера
ENV DATA_PATH=/app/data/heart.csv
ENV MODEL_DIR=/app/models
ENV MLFLOW_URI=/app/mlruns
ENV EXPERIMENT=heart-disease-docker

# ── Рабочая директория ────────────────────────────────────────────────────────
WORKDIR /app

# ── Установка системных зависимостей ─────────────────────────────────────────
# Устанавливаем только необходимое, очищаем кэш apt в одном слое
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        && rm -rf /var/lib/apt/lists/*

# ── Установка Python-зависимостей ────────────────────────────────────────────
# Копируем requirements отдельным слоем — Docker кэширует его,
# если requirements.txt не изменился
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Копирование кода и данных ─────────────────────────────────────────────────
COPY app/train_docker.py ./train_docker.py
COPY data/heart.csv      ./data/heart.csv

# ── Создаём директории для артефактов ────────────────────────────────────────
RUN mkdir -p /app/models /app/mlruns

# ── Порт (для mlflow ui, если нужно) ─────────────────────────────────────────
EXPOSE 5000

# ── Entrypoint — запускаем обучение ──────────────────────────────────────────
CMD ["python", "train_docker.py"]
