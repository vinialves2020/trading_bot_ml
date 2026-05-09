import ccxt
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.database import DatabaseManager

def verificar_resultados():
    db = DatabaseManager('data/trading_data.db')
    exchange = ccxt.binance()
    
    # 1. Carrega apenas trades que ainda estão abertos
    query = "SELECT * FROM trade_history WHERE status = 'OPEN'"
    df_open = pd.read_sql(query, db.conn)
    
    if df_open.empty:
        print("📭 Nenhum trade aberto para auditar.")
        return

    # 3. Exibe o Saldo Acumulado
    exibir_balanco(db)

def exibir_balanco(db):
    df_all = db.load_data('trade_history')
    wins = len(df_all[df_all['status'] == 'WIN'])
    losses = len(df_all[df_all['status'] == 'LOSS'])
    
    # Cálculo considerando sua estratégia 3:1 (Ganha 0.9%, Perde 0.3%)
    # Simulando com banca de $1000
    lucro_total = (wins * 0.001) - (losses * 0.001)
    
    print("\n" + "="*30)
    print(f"📊 DESEMPENHO REAL (SHADOW)")
    print("="*30)
    print(f"🏆 Vitórias: {wins}")
    print(f"📉 Derrotas: {losses}")
    print(f"💵 Lucro Estimado: {lucro_total*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    verificar_resultados()