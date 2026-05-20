"""
Pipeline completo: Baixa dados, cria features e salva no banco.
Executa: fetcher -> features -> salva no SQLite
"""
import sys
import os

sys.path.append(os.path.abspath('.'))

from src.data_pipeline.fetcher import BinanceDataFetcher
from src.data_pipeline.features import FeatureEngineer
from src.data_pipeline.database import DatabaseManager

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Ex: BTC/USDT, ETH/USDT')
    parser.add_argument('--timeframe', type=str, default='15m', help='Ex: 15m, 1h')
    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.timeframe
    prefix = f"{symbol.split('/')[0].lower()}_{timeframe}"
    
    print("=" * 60)
    print(f" PIPELINE COMPLETO: {symbol} no timeframe {timeframe}")
    print("=" * 60)

    print("\n[1/3] Baixando dados da Binance...")
    fetcher = BinanceDataFetcher(symbol=symbol, timeframe=timeframe)
    # Baixa histórico maior para gráfico de 1h
    start_date = "2023-01-01 00:00:00" if timeframe == '1h' else "2024-01-01 00:00:00"
    df_raw = fetcher.fetch_deep_history(start_date_str=start_date)

    if df_raw is None or len(df_raw) == 0:
        print(" Erro: Nao foi possivel baixar dados.")
        return

    print(f" Dados baixados: {len(df_raw)} candles")

    db = DatabaseManager('data/trading_data.db')
    db.save_data(df_raw, f"{prefix}_raw", if_exists='replace')

    print("\n[2/3] Calculando indicadores tecnicos...")
    df_features = FeatureEngineer.apply_indicators(df_raw.copy())

    if timeframe == '1h':
        # 1H precisa de alvos maiores e menos candles de horizonte (8 candles = 8 horas)
        horizon = 8
        tp = 0.015 # 1.5%
        sl = 0.0075 # 0.75%
    else:
        horizon = 16
        tp = 0.004
        sl = 0.002

    df_features = FeatureEngineer.create_target(df_features, horizon=horizon, profit_target=tp, stop_loss=sl)

    print(f" Features calculadas: {len(df_features.columns)} colunas")
    print(f" Linhas prontas: {len(df_features)}")

    db.save_data(df_features, f"{prefix}_features", if_exists='replace')

    print("\n[3/3] Salvando no banco...")
    print(f" Tabela '{prefix}_features' criada com sucesso")

    print("\n" + "="*60)
    print(" PIPELINE CONCLUIDO!")
    print("="*60)
    print(f"\n Agora rode: python src/models/train_xgb.py --symbol {symbol} --timeframe {timeframe}")

if __name__ == "__main__":
    main()
