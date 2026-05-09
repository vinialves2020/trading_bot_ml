import streamlit as st
import pandas as pd
import sqlite3
import json
import os

# Configuração da página
st.set_page_config(page_title="Oráculo BTC | Dashboard", layout="wide")
st.title("📊 Oráculo BTC - Trade Journal & Analytics")
st.sidebar.subheader("🔍 Debug de Infra")
if os.path.exists("/app/data"):
    st.sidebar.write("Arquivos na pasta data:", os.listdir("/app/data"))
else:
    st.sidebar.error("Pasta /app/data não encontrada!")

# Caminhos das fontes de dados
DB_PATH = "/app/data/trading_data.db"
JSONL_PATH = "/app/data/trade_journal.jsonl"

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
    
    # Separa apenas os eventos de fechamento (onde o dinheiro realmente troca de mãos)
    df_closed = df[df['event'] == 'CLOSE'].copy()
    
    # Métricas de Cabeçalho
    col1, col2, col3 = st.columns(3)
    col1.metric("Sinais de Entrada", len(df[df['event'] == 'ENTRY']))
    
    if not df_closed.empty and 'profit_usdt' in df_closed.columns:
        total_pnl = df_closed['profit_usdt'].sum()
        wins = len(df_closed[df_closed['result'] == 'TP'])
        losses = len(df_closed[df_closed['result'] == 'SL'])
        win_rate = (wins / len(df_closed)) * 100 if len(df_closed) > 0 else 0
        
        # Formata com cor condicional nativa do Streamlit
        col2.metric("Lucro/Prejuízo (PnL)", f"${total_pnl:.2f}")
        col3.metric("Taxa de Acerto", f"{win_rate:.1f}%")

        # Gráfico de Evolução do Patrimônio
        st.subheader("📈 Curva de Patrimônio (Paper Trading)")
        if 'paper_balance_after' in df_closed.columns:
            # Plota o saldo oficial calculado pelo bot
            chart_data = df_closed[['timestamp', 'paper_balance_after']].set_index('timestamp')
            st.line_chart(chart_data)
        else:
            # Fallback: soma os lucros
            df_closed['equity_curve'] = df_closed['profit_usdt'].cumsum()
            chart_data = df_closed[['timestamp', 'equity_curve']].set_index('timestamp')
            st.line_chart(chart_data)
    else:
        col2.metric("Lucro/Prejuízo (PnL)", "$0.00")
        col3.metric("Taxa de Acerto", "N/A")
        st.info("Aguardando o fechamento do primeiro trade para calcular o lucro.")

    # Tabela de Trades Detalhada (Mostra tudo: Entries e Closes)
    st.subheader("📝 Diário de Bordo (Trade Journal)")
    
    # Limpa as colunas vazias para deixar a tabela mais bonita
    df_display = df.dropna(axis=1, how='all').sort_index(ascending=False)
    st.dataframe(df_display, use_container_width=True)