import pandas as pd
from src.data_pipeline.database import DatabaseManager

db = DatabaseManager('data/trading_data.db')
df_trades = db.load_data('trade_history')

if df_trades is not None and not df_trades.empty:
    print("\n=== RELATÓRIO DE SHADOW TRADING ===")
    print(f"Total de sinais disparados: {len(df_trades)}")
    print(df_trades.tail(5)) # Mostra os últimos 5 trades
else:
    print("Nenhum trade realizado ainda.")