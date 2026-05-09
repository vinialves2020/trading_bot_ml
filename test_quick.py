"""
Teste rapido para validar:
1. Kelly funcionando (2% minimo)
2. SQLite salvando trade_history
3. JSONL salvando trade_journal.jsonl
"""
import sys
import os
import json
sys.path.append(os.path.abspath('.'))

from src.bot_executor import TradingBot

# Cria instancia sem rodar o loop infinito
bot = TradingBot(threshold=0.95, paper_trading=True)

# Teste 1: Kelly fraction
print("=" * 50)
print("TESTE 1: Kelly Fraction")
print("=" * 50)
print(f" Kelly fraction: {bot.kelly_fraction*100:.2f}%")
print(f" Max risk per trade: {bot.max_risk_per_trade*100:.2f}%")
if bot.max_risk_per_trade > 0:
    print(" Kelly OK (bot vai operar)")
else:
    print(" ERRO: Kelly zero!")

# Teste 2: Calculo de posicao (2% do saldo simulado)
print("\n" + "="*50)
print("TESTE 2: Calculo de Posicao (2% risco)")
print("="*50)
qty = bot._calculate_position_size(80000.0, 79600.0)  # 0.5% SL
valor_usdt = qty * 80000.0
print(f" Saldo: ${bot.paper_balance:.2f}")
print(f" Risco: {bot.max_risk_per_trade*100:.2f}% = ${bot.paper_balance * bot.max_risk_per_trade:.2f}")
print(f" Qty BTC: {qty:.6f} BTC")
print(f" Valor: ${valor_usdt:.2f}")

# Teste 3: Simular entrada e salvar no SQLite + JSONL
print("\n" + "="*50)
print("TESTE 3: Simulando Trade (SQLite + JSONL)")
print("="*50)
bot.open_order = {
    'side': 'LONG',
    'entry_price': 80000.0,
    'take_profit': 80320.0,  # 0.4%
    'stop_loss': 79840.0,     # 0.2%
    'qty_btc': qty,
    'position_size_usdt': valor_usdt,
    'confidence': 0.95,
    'timestamp': __import__('datetime').datetime.now()
}

# Salva no SQLite
try:
    from datetime import datetime
    bot.db_manager.execute_query(
        "INSERT INTO trade_history (timestamp, side, entry_price, take_profit, stop_loss, confidence, position_size_usdt) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(), 'LONG', 80000.0, 80320.0, 79840.0, 0.95, valor_usdt)
    )
    print(" SQLite: Trade salvo com sucesso!")
except Exception as e:
    print(f" SQLite ERRO: {e}")

# Salva no JSONL
try:
    journal_entry = {
        'event': 'ENTRY',
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'side': 'LONG',
        'entry_price': 80000.0,
        'position_size_usdt': valor_usdt,
        'qty_btc': qty
    }
    with open(bot.journal_file, 'a') as f:
        f.write(json.dumps(journal_entry) + '\n')
    print(f" JSONL: Trade salvo em {bot.journal_file}")
except Exception as e:
    print(f" JSONL ERRO: {e}")

# Teste 4: Verifica se salvou no SQLite
print("\n" + "="*50)
print("TESTE 4: Lendo SQLite e JSONL")
print("="*50)
try:
    df = bot.db_manager.load_data('trade_history')
    print(f" SQLite: {len(df)} trades na tabela")
    if len(df) > 0:
        print(df.tail(2))
except Exception as e:
    print(f" SQLite leitura ERRO: {e}")

try:
    with open(bot.journal_file) as f:
        lines = f.readlines()
    print(f" JSONL: {len(lines)} linhas no arquivo")
    if lines:
        print(f" Ultima linha: {lines[-1][:100]}...")
except Exception as e:
    print(f" JSONL leitura ERRO: {e}")

print("\n" + "="*50)
print("TESTE CONCLUIDO!")
print("="*50)
