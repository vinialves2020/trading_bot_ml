FROM python:3.11-slim

# Adicionamos 'git' à lista de instalação do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala o Torch CPU (mantemos essa otimização para economizar RAM)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# Instala o resto das dependências
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/bot_executor.py", "--paper"]