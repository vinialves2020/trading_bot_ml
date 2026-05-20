"""
seed_fake_trades.py
Injeta trades simulados para testar o dashboard.
Para rodar na EC2:
    python scripts_agentes/seed_fake_trades.py

Para apagar os dados falsos depois:
    python scripts_agentes/seed_fake_trades.py --limpar
"""

import sqlite3
import json
import os
import sys
from datetime import datetime, timezone, timedelta
import random

DB_PATH    = "data/trading_data.db"
JSONL_PATH = "data/trade_journal.jsonl"

MOEDAS = [
    {"symbol": "BTC/USDT", "timeframe": "15m", "base_price": 67000.0},
    {"symbol": "ETH/USDT", "timeframe": "1h",  "base_price": 3500.0},
    {"symbol": "SOL/USDT", "timeframe": "15m", "base_price": 170.0},
]

def criar_tabela(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_history (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol              TEXT,
            timeframe           TEXT,
            timestamp           DATETIME,
            side                TEXT,
            entry_price         REAL,
            take_profit         REAL,
            stop_loss           REAL,
            confidence          REAL,
            position_size_usdt  REAL,
            result              TEXT,
            profit_pct          REAL,
            profit_usdt         REAL,
            paper_balance_after REAL,
            event               TEXT DEFAULT 'ENTRY'
        )
    """)
    conn.commit()

    # Migração: adiciona colunas novas se o banco já existia com schema antigo
    colunas_existentes = {row[1] for row in conn.execute("PRAGMA table_info(trade_history)")}
    migracoes = {
        "symbol":              "TEXT",
        "timeframe":           "TEXT",
        "result":              "TEXT",
        "profit_pct":          "REAL",
        "profit_usdt":         "REAL",
        "paper_balance_after": "REAL",
        "event":               "TEXT DEFAULT 'ENTRY'",
    }
    for col, tipo in migracoes.items():
        if col not in colunas_existentes:
            conn.execute(f"ALTER TABLE trade_history ADD COLUMN {col} {tipo}")
            print(f"  ↳ Migração: coluna '{col}' adicionada.")
    conn.commit()

def seed(n_trades_por_moeda=6):
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    criar_tabela(conn)

    journal_lines = []
    now = datetime.now(timezone.utc)

    for moeda in MOEDAS:
        sym       = moeda["symbol"]
        tf        = moeda["timeframe"]
        preco     = moeda["base_price"]
        saldo     = 100.0

        print(f"\n🔧 Gerando {n_trades_por_moeda} trades para {sym}...")

        for i in range(n_trades_por_moeda):
            ts = now - timedelta(hours=(n_trades_por_moeda - i) * 2)

            # Parâmetros do trade
            atr         = preco * random.uniform(0.003, 0.007)
            entry       = round(preco * random.uniform(0.998, 1.002), 4)
            sl          = round(entry - atr * 1.5, 4)
            tp          = round(entry + atr * 3.0, 4)
            confidence  = round(random.uniform(0.61, 0.89), 4)
            pos_usdt    = round(saldo * 0.02, 4)
            qty         = pos_usdt / entry

            # Resultado: 60% TP, 40% SL (simula modelo razoável)
            result       = "TP" if random.random() < 0.60 else "SL"
            exit_price   = tp if result == "TP" else sl
            gross        = qty * (exit_price - entry)
            fee          = (qty * entry * 0.001) + (qty * exit_price * 0.001)
            profit_usdt  = round(gross - fee, 4)
            profit_pct   = round((profit_usdt / pos_usdt) * 100, 4)
            saldo        = round(saldo + profit_usdt, 4)

            # ── SQLite: ENTRY ──────────────────────────────────────────────
            conn.execute("""
                INSERT INTO trade_history
                  (symbol, timeframe, timestamp, side, entry_price, take_profit,
                   stop_loss, confidence, position_size_usdt, result,
                   profit_pct, profit_usdt, paper_balance_after, event)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'CLOSE')
            """, (sym, tf, ts.isoformat(), "LONG", entry, tp, sl,
                  confidence, pos_usdt, result, profit_pct, profit_usdt, saldo))

            # ── JSONL ──────────────────────────────────────────────────────
            journal_lines.append({
                "event": "CLOSE",
                "symbol": sym,
                "timeframe": tf,
                "timestamp": ts.isoformat(),
                "side": "LONG",
                "entry_price": entry,
                "exit_price": exit_price,
                "take_profit": tp,
                "stop_loss": sl,
                "result": result,
                "confidence": confidence,
                "profit_pct": profit_pct,
                "profit_usdt": profit_usdt,
                "paper_balance_after": saldo,
                "position_size_usdt": pos_usdt,
            })

            emoji = "✅" if result == "TP" else "❌"
            print(f"  {emoji} Trade {i+1}: {result} | PnL: ${profit_usdt:+.2f} | Saldo: ${saldo:.2f}")

    conn.commit()
    conn.close()

    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        for line in journal_lines:
            f.write(json.dumps(line) + "\n")

    print(f"\n✅ Concluído! {len(journal_lines)} trades inseridos.")
    print(f"   → SQLite : {DB_PATH}")
    print(f"   → JSONL  : {JSONL_PATH}")
    print("\n💡 Para apagar: python scripts_agentes/seed_fake_trades.py --limpar")

def limpar():
    """Apaga TODOS os dados (fake e reais). Use com cuidado."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM trade_history")
    conn.commit()
    conn.close()
    # Limpa o JSONL
    open(JSONL_PATH, "w").close()
    print("🗑️  Banco de dados e JSONL limpos com sucesso.")

if __name__ == "__main__":
    if "--limpar" in sys.argv:
        confirm = input("⚠️  Isso apaga TODOS os trades do banco. Confirma? (s/N): ")
        if confirm.strip().lower() == "s":
            limpar()
        else:
            print("Operação cancelada.")
    else:
        seed()
