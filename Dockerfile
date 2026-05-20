# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# BuildKit cache mount: o pip guarda os pacotes baixados entre builds.
# Torch (~200MB) só é baixado na PRIMEIRA vez. Builds seguintes: segundos.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY . .

CMD ["python", "src/bot_executor.py", "--paper"]