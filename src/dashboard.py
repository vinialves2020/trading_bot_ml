import streamlit as st
import pandas as pd
import sqlite3
import os

# Configuração da página
st.set_page_config(page_title="Oráculo BTC | Dashboard", layout="wide")
st.title("📊 Oráculo BTC - Monitoramento ao Vivo")

# Caminho do banco de dados (ajuste se o seu bot salvar com outro nome)
DB_PATH = "data/trades.db"

def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    # Conecta no SQLite e puxa o histórico
    conn = sqlite3.connect(DB_PATH)
    # Supondo que sua tabela se chame 'trades'. Ajuste se for diferente!
    try:
        df = pd.read_sql_query("SELECT * FROM trades", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

df = load_data()

if df.empty:
    st.warning("Aguardando o bot registrar o primeiro trade no banco de dados...")
else:
    # Mostra os últimos trades em uma tabela bonita
    st.subheader("Últimas Operações")
    st.dataframe(df.tail(10).sort_index(ascending=False))

    # Se você tiver uma coluna de lucro (ex: 'pnl' ou 'profit'), podemos plotar o gráfico de saldo
    if 'pnl' in df.columns:
        st.subheader("Curva de Patrimônio (PnL)")
        df['pnl_acumulado'] = df['pnl'].cumsum()
        st.line_chart(df['pnl_acumulado'])

    # Métricas rápidas
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Trades", len(df))
    # Adapte as colunas abaixo conforme o nome exato que o seu bot salva no SQLite
    if 'status' in df.columns:
        wins = len(df[df['status'] == 'WIN'])
        col2.metric("Vitórias", wins)
        col3.metric("Taxa de Acerto", f"{(wins/len(df))*100:.1f}%" if len(df) > 0 else "0%")