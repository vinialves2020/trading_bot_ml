#!/bin/bash

# ==============================================================================
# SCRIPT DE DEPLOY E OTIMIZACAO PARA AWS EC2 (t2.micro - Ubuntu)
# ==============================================================================
# Este script configura a maquina para suportar multiplos containers de IA,
# resolvendo o problema de memoria RAM limitada (1GB) da instancia t2.micro.

echo "🚀 Iniciando configuracao do servidor AWS (DevOps Agent)..."

# 1. Atualizar pacotes
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. Criar Arquivo de Swap (Memoria Virtual)
# Uma instancia t2.micro tem apenas 1GB de RAM. Os 4 containers (FinBERT, BTC, ETH, SOL)
# consomem cerca de 2.5GB no pico de carregamento. O Swap impede o servidor de travar.
echo "💾 Configurando Swap de 3GB para evitar OOM (Out-Of-Memory) Crash..."
if [ -f /swapfile ]; then
    echo "Swap ja configurado."
else
    sudo fallocate -l 3G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    
    # Otimiza a frequencia com que o Ubuntu usa o Swap (Swappiness)
    # Valor padrao e 60, reduzimos para 10 para tentar manter os bots em memoria fisica ao maximo
    sudo sysctl vm.swappiness=10
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    echo "Swap configurado com sucesso!"
fi

# 3. Instalar Docker e Docker Compose
if ! command -v docker &> /dev/null; then
    echo "🐳 Instalando Docker..."
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    sudo mkdir -m 0755 -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Permissoes para rodar docker sem sudo (requer re-login, mas garantimos com sg abaixo)
    sudo usermod -aG docker ubuntu
fi

# 4. Criar rede Docker e Iniciar
echo "🏗️ Construindo a Imagem Base (isso pode demorar na primeira vez)..."
sudo docker build -t oraculo_base_image:latest .

echo "🔥 Subindo o Enxame (Swarm) de Bots HFT..."
sudo docker compose up -d

echo "=============================================================================="
echo "✅ DEPLOY CONCLUIDO COM SUCESSO!"
echo "=============================================================================="
echo "O Dashboard do Streamlit estara disponivel na porta 80 do seu IP publico AWS."
echo "Para visualizar os logs de todos os robos simultaneamente, digite:"
echo "sudo docker compose logs -f"
