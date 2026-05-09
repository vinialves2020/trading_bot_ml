import streamlit as st
import pandas as pd
import sqlite3
import json
import os

# Configuração da página
st.set_page_config(page_title="Oráculo BTC | Dashboard", layout="wide")
st.title("📊 Oráculo BTC - Trade Journal & Analytics")

# Caminhos das fontes de dados
DB_PATH = "data/trading_data.db"
JSONL_PATH = "data/trade_journal.jsonl"

def load_data():
    # 1. Tenta carregar o JSONL (Mais detalhado)
    if os.path.exists(JSONL_PATH):
        try:
            with open(JSONL_PATH, 'r') as f:
                data = [json.loads(line) for line in f]
            df_json = pd.DataFrame(data)
            if not df_json.empty:
                return df_json, "JSONL (Detalhado)"
        except Exception as e:
            st.error(f"Erro ao ler JSONL: {e}")

    # 2. Backup: Tenta carregar do SQLite (Tabela: trade_history)
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            # Ajustado para a tabela correta informada
            df_db = pd.read_sql_query("SELECT * FROM trade_history", conn)
            conn.close()
            if not df_db.empty:
                return df_db, "SQLite (Histórico)"
        except Exception as e:
            st.error(f"Erro ao ler SQLite: {e}")
            
    return pd.DataFrame(), None

df, source_name = load_data()

if df.empty:
    st.warning("Aguardando o bot registrar operações nas fontes de dados...")
else:
    st.sidebar.success(f"Fonte: {source_name}")
    
    # Métricas de Cabeçalho
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Trades", len(df))
    
    if 'pnl' in df.columns:
        total_pnl = df['pnl'].sum()
        col2.metric("PnL Acumulado", f"${total_pnl:.2f}", delta=f"{total_pnl:.2f}")
        
    if 'type' in df.columns:
        longs = len(df[df['type'] == 'LONG'])
        col3.metric("Operações Long", longs)

    # Tabela de Trades Detalhada
    st.subheader("📝 Diário de Bordo (Trade Journal)")
    # Reordenando para ver o mais recente primeiro
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)

    # Gráfico de Evolução
    if 'pnl' in df.columns:
        st.subheader("📈 Curva de Patrimônio")
        df['equity_curve'] = df['pnl'].cumsum()
        st.line_chart(df['equity_curve'])