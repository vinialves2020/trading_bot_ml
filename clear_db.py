import sqlite3
import os

db_path = os.path.expanduser('~/trading_bot_ml/data/trading_data.db')
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute('DELETE FROM trade_history')
        conn.commit()
        print("Tabela trade_history limpa com sucesso na AWS.")
    except Exception as e:
        print(f"Erro: {e}")
    conn.close()

journal_path = os.path.expanduser('~/trading_bot_ml/data/trade_journal.jsonl')
if os.path.exists(journal_path):
    with open(journal_path, 'w') as f:
        pass
    print("trade_journal.jsonl limpo com sucesso.")
