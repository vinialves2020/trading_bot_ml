FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Unificamos o RUN: Instala o Torch CPU e, na mesma camada, instala o resto.
# Isso impede o pip de baixar a versão da NVIDIA por engano.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/bot_executor.py", "--paper"]