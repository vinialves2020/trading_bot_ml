import sqlite3
import pandas as pd

def relatorio_risco():
    conn = sqlite3.connect('data/trading_data.db')
    try:
        df = pd.read_sql("SELECT * FROM paper_trades", conn)
        print(f"Total de Trades: {len(df)}")
        print(f"Lucro Total (PnL): {df['profit_usdt'].sum()}")
        print(f"Win Rate: {(len(df[df['result']=='TP']) / len(df)) * 100:.2f}%")
    except Exception as e:
        print("Tabela ainda vazia ou sem operações fechadas.")
    finally:
        conn.close()

if __name__ == "__main__":
    relatorio_risco()