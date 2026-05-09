import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.database import DatabaseManager
from src.data_pipeline.features import FeatureEngineer
from xgboost import XGBClassifier
import vectorbt as vbt

def main():
    print(" Backtest Profissional - XGBoost BTC/USDT 15m (VectorBT)")
    print("=" * 60)

    # 1. Carregar dados do banco
    db = DatabaseManager('data/trading_data.db')

    # Tenta carregar dados com features
    df = db.load_data('btc_15m_master')
    if df is None:
        df = db.load_data('btc_15m_features')
    if df is None:
        df = db.load_data('btc_15m_ml_dataset')

    if df is None or len(df) == 0:
        print(" Erro: Sem dados no banco. Execute o fetcher primeiro.")
        return

    print(f" Dados carregados: {len(df)} candles")
    print(f" Periodo: {df.index[0]} a {df.index[-1]}")

    # 2. Carregar modelo XGBoost
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    model_path = os.path.join(base_path, "data", "models_weights", "xgb_oraculo_btc.json")

    if not os.path.exists(model_path):
        print(f" Modelo nao encontrado em: {model_path}")
        return

    model = XGBClassifier()
    model.load_model(model_path)
    features_list = FeatureEngineer.get_feature_list()

    # 3. Walk-Forward: Treino 80% / Teste 20%
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()

    print(f"\n Walk-Forward Analysis")
    print(f" Teste (Out-of-Sample): {df_test.index[0]} a {df_test.index[-1]}")
    print(f" Candles no teste: {len(df_test)}")

    # 4. Gerar sinais com XGBoost
    print("\n Gerando sinais com XGBoost...")

    # Preparar features
    X_test = df_test[features_list].fillna(0)

    # Predicao
    prob = model.predict_proba(X_test)[:, 1]  # Probabilidade de alta
    threshold = 0.50
    signals = (prob >= threshold).astype(int)

    # Para scalping: 1 = LONG, 0 = NEUTRO
    entries = (signals == 1) & (signals.shift(1) == 0)
    exits = (signals == 0) & (signals.shift(1) == 1)

    print(f" Sinais gerados: {signals.sum()} LONG em {len(signals)} candles")
    print(f" Entradas: {entries.sum()} | Saidas: {exits.sum()}")

    # 5. Executar Backtest Vetorizado (VectorBT)
    print("\n Executando backtest vetorizado...")

    close_prices = df_test['close'].values

    # Usar VectorBT para simulacao
    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=entries.values,
        exits=exits.values,
        fees=0.001,  # Taxa Binance 0.1%
        init_cash=10000.0,
        size=0.02,  # 2% do capital por trade (prompt.md)
        size_type='percent',
        freq='15T'
    )

    # 6. Metricas de Performance (prompt.md linha 22)
    print("\n" + "=" * 60)
    print(" METRICAS DE PERFORMANCE")
    print("=" * 60)

    total_return = portfolio.total_return()
    sharpe = portfolio.sharpe_ratio()
    sortino = portfolio.sortino_ratio()
    max_dd = portfolio.max_drawdown()

    print(f" Retorno Total: {total_return*100:.2f}%")
    print(f" Sharpe Ratio: {sharpe:.2f}")
    print(f" Sortino Ratio: {sortino:.2f}")
    print(f" Max Drawdown: {max_dd*100:.2f}%")

    # Trades (VectorBT: usar records para acesso seguro)
    trades = portfolio.trades
    if trades.count > 0:
        # Extrair records como DataFrame/array
        records = trades.records

        # VectorBT retorna records como numpy recarray ou DataFrame
        if hasattr(records, 'pnl'):
            pnl_values = records.pnl
        else:
            # Fallback para .values ou lista
            pnl_values = list(records) if records is not None else []

        if len(pnl_values) > 0:
            pnl_array = np.array(pnl_values)
            winning = pnl_array[pnl_array > 0]
            losing = pnl_array[pnl_array <= 0]

            win_rate = (len(winning) / len(pnl_array)) * 100
            avg_win = winning.mean() if len(winning) > 0 else 0
            avg_loss = losing.mean() if len(losing) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

            print(f"\n Trades Totais: {len(pnl_array)}")
            print(f" Win Rate: {win_rate:.1f}%")
            print(f" Avg Win: ${avg_win:.2f}")
            print(f" Avg Loss: ${avg_loss:.2f}")
            print(f" Profit Factor: {profit_factor:.2f}")
        else:
            print(f"\n Trades detectados: {trades.count} mas sem PnL disponivel")

    # 7. Salvar resultados em JSON (prompt.md - Trade Journaling)
    results = {
        'timestamp': datetime.now().isoformat(),
        'period': {
            'start': str(df_test.index[0]),
            'end': str(df_test.index[-1]),
            'candles': len(df_test)
        },
        'model': 'xgb_oraculo_btc.json',
        'threshold': threshold,
        'metrics': {
            'total_return_pct': float(total_return * 100),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown_pct': float(max_dd * 100),
            'total_trades': int(len(trades)) if len(trades) > 0 else 0,
            'win_rate_pct': float(win_rate) if len(trades) > 0 else 0,
            'profit_factor': float(profit_factor) if len(trades) > 0 else 0
        },
        'walk_forward': {
            'train_period': f"{df.index[0]} to {df.index[split_idx]}",
            'test_period': f"{df.index[split_idx]} to {df.index[-1]}"
        }
    }

    results_path = os.path.join(base_path, "data", "backtest_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n Resultados salvos em: {results_path}")

    # 8. Gerar grafico (VectorBT)
    try:
        fig = portfolio.plot()
        chart_path = os.path.join(base_path, "data", "backtest_chart.html")
        fig.write_html(chart_path)
        print(f" Grafico salvo em: {chart_path}")
    except Exception as e:
        print(f" Aviso: Nao foi possivel gerar grafico: {e}")

    print("\n Backtest concluido!")

if __name__ == "__main__":
    main()
