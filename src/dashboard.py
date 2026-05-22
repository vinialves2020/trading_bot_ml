import streamlit as st
import pandas as pd
import sqlite3
import json
import os

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oráculo Quant | Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Oráculo BTC — Trade Journal & Analytics")

DB_PATH    = "/app/data/trading_data.db"
JSONL_PATH = "/app/data/trade_journal.jsonl"
LOG_DIR    = "/app/data/logs"

COIN_EMOJI = {
    "BTC/USDT": "₿ BTC/USDT",
    "ETH/USDT": "Ξ ETH/USDT",
    "SOL/USDT": "◎ SOL/USDT",
}

LOG_FILES = {
    "₿ BTC/USDT": "bot_BTC_USDT.log",
    "Ξ ETH/USDT": "bot_ETH_USDT.log",
    "◎ SOL/USDT": "bot_SOL_USDT.log",
}

# ── Carregamento de dados com cache ───────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data():
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT * FROM trade_history ORDER BY id DESC", conn)
            conn.close()
            if not df.empty:
                return df, "SQLite ✅"
        except Exception as e:
            st.sidebar.warning(f"Erro SQLite: {e}")

    if os.path.exists(JSONL_PATH):
        try:
            with open(JSONL_PATH, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
            df = pd.DataFrame(data)
            if not df.empty:
                return df, "JSONL ✅"
        except Exception as e:
            st.sidebar.warning(f"Erro JSONL: {e}")

    return pd.DataFrame(), None

@st.cache_data(ttl=5)
def load_log(filepath, n_lines):
    """Lê as últimas N linhas de um arquivo de log."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except Exception as e:
        return f"Erro ao ler log: {e}"

def calc_metrics(df_closed):
    if df_closed.empty:
        return None
    total_pnl = df_closed["profit_usdt"].sum() if "profit_usdt" in df_closed else 0
    wins      = (df_closed["result"] == "TP").sum() if "result" in df_closed else 0
    losses    = (df_closed["result"] == "SL").sum() if "result" in df_closed else 0
    total     = wins + losses
    win_rate  = (wins / total * 100) if total > 0 else 0
    last_bal  = (df_closed["paper_balance_after"].dropna().iloc[0]
                 if "paper_balance_after" in df_closed and not df_closed["paper_balance_after"].dropna().empty
                 else None)
    return {"total_pnl": total_pnl, "wins": wins, "losses": losses,
            "total": total, "win_rate": win_rate, "last_balance": last_bal}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controles")
with st.sidebar.expander("🔍 Debug de Infra"):
    if os.path.exists("/app/data"):
        st.write("Arquivos em /app/data:", os.listdir("/app/data"))
    else:
        st.error("Pasta /app/data não encontrada!")

df, source_name = load_data()

# ── TABS principais ───────────────────────────────────────────────────────────
tab_trades, tab_logs = st.tabs(["📈 Trades & Performance", "📡 Logs ao Vivo"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRADES & PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_trades:
    if df.empty:
        st.warning("⏳ Aguardando o bot registrar operações...")
    else:
        st.sidebar.success(f"Fonte: {source_name}")

        symbols_available = sorted(df["symbol"].dropna().unique().tolist()) if "symbol" in df.columns else []
        view_options = ["🌐 Portfólio Global (Todos)"] + [COIN_EMOJI.get(s, s) for s in symbols_available]
        selected_view = st.sidebar.selectbox("🔍 Filtro de Moeda", view_options)

        if selected_view == "🌐 Portfólio Global (Todos)":
            df_view = df.copy()
        else:
            selected_symbol = symbols_available[view_options.index(selected_view) - 1]
            df_view = df[df["symbol"] == selected_symbol].copy()

        col_event = "event" if "event" in df_view.columns else None
        df_closed = df_view[df_view[col_event] == "CLOSE"].copy() if col_event else df_view.copy()
        df_entry  = df_view[df_view[col_event] == "ENTRY"].copy() if col_event else pd.DataFrame()

        # Cartões por moeda no modo Global
        if selected_view == "🌐 Portfólio Global (Todos)" and symbols_available:
            st.subheader("📈 Performance por Moeda")
            cols = st.columns(len(symbols_available))
            for i, sym in enumerate(symbols_available):
                df_sym_closed = df_closed[df_closed["symbol"] == sym] if "symbol" in df_closed.columns else pd.DataFrame()
                m = calc_metrics(df_sym_closed)
                label = COIN_EMOJI.get(sym, sym)
                with cols[i]:
                    st.markdown(f"### {label}")
                    if m:
                        st.metric("PnL Líquido",  f"${m['total_pnl']:.2f}")
                        st.metric("Win Rate",      f"{m['win_rate']:.1f}%",
                                  delta=f"{m['wins']}W / {m['losses']}L")
                        if m["last_balance"] is not None:
                            st.metric("Saldo Atual", f"${m['last_balance']:.2f}")
                    else:
                        st.info("Sem trades fechados")
            st.divider()

        # Métricas agregadas
        st.subheader(f"📊 Métricas — {selected_view}")
        m = calc_metrics(df_closed)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sinais de Entrada", len(df_entry) if not df_entry.empty else len(df_view))
        if m:
            col2.metric("PnL Líquido",   f"${m['total_pnl']:.2f}")
            col3.metric("Win Rate",       f"{m['win_rate']:.1f}%",
                        delta=f"{m['wins']}W / {m['losses']}L de {m['total']}")
            if m["last_balance"] is not None:
                col4.metric("Saldo Simulado", f"${m['last_balance']:.2f}")
        else:
            col2.metric("PnL Líquido",    "$0.00")
            col3.metric("Win Rate",       "N/A")
            col4.metric("Saldo Simulado", "Aguardando")

        # Curva de Patrimônio
        st.subheader("📈 Curva de Patrimônio (Paper Trading)")
        if not df_closed.empty:
            if "paper_balance_after" in df_closed.columns and df_closed["paper_balance_after"].notna().any():
                df_chart = df_closed[["timestamp", "paper_balance_after", "symbol"]].dropna().copy()
                df_chart = df_chart.sort_values("timestamp")
                if selected_view == "🌐 Portfólio Global (Todos)" and "symbol" in df_chart.columns:
                    df_pivot = df_chart.pivot_table(index="timestamp", columns="symbol",
                                                    values="paper_balance_after", aggfunc="last")
                    st.line_chart(df_pivot)
                else:
                    st.line_chart(df_chart.set_index("timestamp")[["paper_balance_after"]])
            elif "profit_usdt" in df_closed.columns:
                df_s = df_closed.sort_values("timestamp").copy()
                df_s["equity_curve"] = df_s["profit_usdt"].cumsum() + 100
                st.line_chart(df_s.set_index("timestamp")[["equity_curve"]])
        else:
            st.info("Aguardando o fechamento do primeiro trade para calcular a curva.")

        # Tabela detalhada
        st.subheader("📝 Diário de Bordo Completo")
        col_rename = {
            "symbol": "Moeda", "timeframe": "Tempo Gráfico", "event": "Evento",
            "timestamp": "Data/Hora (UTC)", "side": "Direção",
            "entry_price": "Entrada ($)", "take_profit": "Take Profit ($)",
            "stop_loss": "Stop Loss ($)", "confidence": "Confiança IA",
            "result": "Resultado", "profit_pct": "PnL (%)",
            "profit_usdt": "PnL ($)", "paper_balance_after": "Saldo Após ($)",
            "position_size_usdt": "Tamanho ($)",
        }
        df_display = df_view.rename(columns=col_rename).dropna(axis=1, how="all")
        if "Data/Hora (UTC)" in df_display.columns:
            df_display = df_display.sort_values("Data/Hora (UTC)", ascending=False)
        if "Confiança IA" in df_display.columns:
            def format_conf(x):
                if pd.isna(x): return ""
                if isinstance(x, bytes):
                    try:
                        import numpy as np
                        return f"{np.frombuffer(x, dtype=np.float32)[0]*100:.1f}%"
                    except Exception:
                        return "N/A"
                try:
                    return f"{float(x)*100:.1f}%"
                except Exception:
                    return "N/A"
            df_display["Confiança IA"] = df_display["Confiança IA"].apply(format_conf)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LOGS AO VIVO
# ══════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.subheader("📡 Logs dos Containers em Tempo Real")
    st.caption("Atualização automática a cada 5 segundos. Máximo 512KB por arquivo (rotação automática).")

    # Seletor de container + controles na mesma linha
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        selected_log_label = st.selectbox("🤖 Container", list(LOG_FILES.keys()), label_visibility="collapsed")
    with c2:
        n_lines = st.slider("Últimas linhas", 20, 500, 100, label_visibility="collapsed")
    with c3:
        limpar = st.button("🗑️ Limpar Log", use_container_width=True)

    log_filepath = os.path.join(LOG_DIR, LOG_FILES[selected_log_label])

    # Ação de limpar
    if limpar:
        try:
            open(log_filepath, "w").close()
            st.success(f"✅ Log de {selected_log_label} limpo!")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao limpar: {e}. Verifique permissões do arquivo.")

    # Exibe o log
    log_content = load_log(log_filepath, n_lines)

    if log_content is None:
        st.info(f"⏳ Aguardando o bot {selected_log_label} iniciar e gerar logs em `{log_filepath}`...")
        st.caption("Os logs aparecem automaticamente após o primeiro candle ser processado pelo robô.")
    else:
        # Tamanho atual do arquivo
        try:
            size_kb = os.path.getsize(log_filepath) / 1024
            st.caption(f"📁 Arquivo: `{log_filepath}` | Tamanho: **{size_kb:.1f} KB** | Mostrando últimas **{n_lines}** linhas")
        except Exception:
            pass

        st.code(log_content, language=None)

    # Botão de refresh manual
    if st.button("🔄 Atualizar Logs Agora"):
        st.cache_data.clear()
        st.rerun()