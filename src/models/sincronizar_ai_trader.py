import sqlite3
import json
import requests
import os
from datetime import datetime

# ==========================================
# CONFIGURAÇÕES DO AI-TRADER
# ==========================================
# Insira a URL real da API do AI-Trader (ex: http://IP_DO_SERVIDOR:8000/api/signals ou a URL da plataforma deles)
AI_TRADER_API_URL = os.getenv("AI_TRADER_URL", "https://api.hkuds.ai/v1/signals") 
AI_TRADER_TOKEN = os.getenv("AI_TRADER_TOKEN", "SEU_TOKEN_DE_ACESSO_AQUI")

# ==========================================
# 🛠️ Leitura do Banco de Dados
# ==========================================
def buscar_historico_trades(limite: int = 5) -> list:
    """
    Busca as últimas operações no SQLite.
    Corrigido para ler a tabela real do bot ('trade_history').
    """
    try:
        # Garante que o caminho funcione de qualquer pasta
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        db_path = os.path.join(base_path, 'data', 'trading_data.db')
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Puxa os últimos trades fechados (Ignora os que ainda estão 'OPEN')
        query = f"""
            SELECT timestamp, side, entry_price, take_profit, stop_loss, confidence, status 
            FROM trade_history 
            WHERE status != 'OPEN' 
            ORDER BY timestamp DESC LIMIT {limite}
        """
        cursor.execute(query)
        colunas = [descricao[0] for descricao in cursor.description]
        
        trades = []
        for linha in cursor.fetchall():
            trades.append(dict(zip(colunas, linha)))
            
        conn.close()
        return trades
    except Exception as e:
        print(f"🚨 Erro ao ler banco de dados: {e}")
        return []

# ==========================================
# 📡 Envio Direto via API REST
# ==========================================
def enviar_para_ai_trader(trades):
    if not trades:
        print("Nenhum trade fechado encontrado para sincronizar.")
        return

    # Formatação padrão de payload para Signal Providers
    payload = {
        "agent_name": "Oráculo BTC",
        "strategy": "XGBoost + LightGBM (15m Scalping)",
        "symbol": "BTC/USDT",
        "trades": trades,
        "sync_timestamp": datetime.utcnow().isoformat() + "Z"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI_TRADER_TOKEN}"
    }

    print(f"🌐 Empacotando {len(trades)} trades para envio ao AI-Trader...")
    print(json.dumps(payload, indent=2))
    
    try:
        print(f"\n📡 Enviando POST para {AI_TRADER_API_URL}...")
        
        # Descomente a linha abaixo quando tiver a URL e o Token corretos:
        # response = requests.post(AI_TRADER_API_URL, json=payload, headers=headers, timeout=10)
        # 
        # if response.status_code in [200, 201]:
        #     print("✅ Sincronização concluída com sucesso na rede AI-Trader!")
        # else:
        #     print(f"⚠️ Erro da API ({response.status_code}): {response.text}")
        
        print("✅ [MOCK] Integração concluída! (Descomente o requests.post no código quando tiver sua URL/Token).")
        
    except Exception as e:
        print(f"🚨 Falha de conexão com a API do AI-Trader: {e}")

if __name__ == "__main__":
    ultimos_trades = buscar_historico_trades(limite=5)
    enviar_para_ai_trader(ultimos_trades)