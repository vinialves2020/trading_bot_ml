FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala primeiro o Torch CPU separadamente (Evita baixar 4GB de NVIDIA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# Instala o resto ignorando o torch que já está lá
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/bot_executor.py", "--paper"]