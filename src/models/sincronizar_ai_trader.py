import sqlite3
import json
import requests
import os

# ==========================================
# CONFIGURAÇÕES DO AI-TRADER
# ==========================================
# URL base oficial da documentação ai4trade.ai
AI_TRADER_API_URL = "https://ai4trade.ai/api/signals/realtime"
# Chave da sua conta "OraculoBTC"
AI_TRADER_TOKEN = "HyCq0jy90uXxNJhuqzUmHBiqeKql0oIvNbNE_qPDyWE"

# ==========================================
# 🛠️ Leitura do Banco de Dados
# ==========================================
def buscar_historico_trades(limite: int = 5) -> list:
    try:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        db_path = os.path.join(base_path, 'data', 'trading_data.db')
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
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
# 📡 Envio de Sinais (Sync External Trade)
# ==========================================
def enviar_para_ai_trader(trades):
    if not trades:
        print("Nenhum trade fechado encontrado para sincronizar.")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI_TRADER_TOKEN}"
    }

    print(f"🌐 Sincronizando {len(trades)} trades com o servidor AI-Trader...")
    
    for trade in trades:
        # Formato de Sincronização Externa (Method 1: Sync External Trade) conforme SKILL.md
        payload = {
            "market": "crypto",
            "action": "buy" if trade["side"] == "LONG" else "short",
            "symbol": "BTC",
            "price": trade["entry_price"],
            "quantity": 0.05, # Quantidade fixa para padronização de ranking na rede
            "content": f"🎯 Alvo (TP): {trade['take_profit']} | 🛡️ Stop (SL): {trade['stop_loss']} | Confiança IA: {trade['confidence']:.2%}\n[Resultado do Oráculo: {trade['status']}]",
            "executed_at": trade["timestamp"]
        }

        try:
            print(f"📡 Enviando sinal do trade de {trade['timestamp']}...")
            response = requests.post(AI_TRADER_API_URL, json=payload, headers=headers, timeout=10)
            
            if response.status_code in [200, 201]:
                print(f"✅ Trade sincronizado com sucesso!")
            else:
                print(f"⚠️ Erro da API ({response.status_code}): {response.text}")
                
        except Exception as e:
            print(f"🚨 Falha de conexão: {e}")

if __name__ == "__main__":
    ultimos_trades = buscar_historico_trades(limite=5)
    enviar_para_ai_trader(ultimos_trades)