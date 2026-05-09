# Usa uma imagem leve do Python
FROM python:3.11-slim

# Instala dependências de sistema necessárias para compilar XGBoost e LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia apenas o arquivo de requisitos primeiro (otimiza o cache do Docker)
COPY requirements.txt .

# Instala as bibliotecas Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código do projeto
COPY . .

# Comando para rodar o bot (por padrão inicia em modo Paper Trading)
CMD ["python", "src/bot_executor.py", "--paper"]