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

def main():
    print("=" * 60)
    print(" PIPELINE COMPLETO: DADOS -> FEATURES -> BANCO")
    print("=" * 60)

    # 1. Baixar dados
    print("\n[1/3] Baixando dados da Binance...")
    fetcher = BinanceDataFetcher(symbol='BTC/USDT', timeframe='15m')
    df_raw = fetcher.fetch_deep_history(start_date_str="2024-01-01 00:00:00")

    if df_raw is None or len(df_raw) == 0:
        print(" Erro: Nao foi possivel baixar dados.")
        return

    print(f" Dados baixados: {len(df_raw)} candles")

    # 2. Salvar raw no banco
    db = DatabaseManager('data/trading_data.db')
    db.save_data(df_raw, 'btc_15m_raw', if_exists='replace')

    # 3. Criar features
    print("\n[2/3] Calculando indicadores tecnicos...")
    df_features = FeatureEngineer.apply_indicators(df_raw.copy())

    # Criar target (Triple Barrier alinhado)
    df_features = FeatureEngineer.create_target(df_features, horizon=16, profit_target=0.004, stop_loss=0.002)

    print(f" Features calculadas: {len(df_features.columns)} colunas")
    print(f" Linhas prontas: {len(df_features)}")

    # 4. Salvar com features no banco
    db.save_data(df_features, 'btc_15m_features', if_exists='replace')

    print("\n[3/3] Salvando no banco...")
    print(f" Tabela 'btc_15m_features' criada com {len(df_features)} linhas")

    print("\n" + "="*60)
    print(" PIPELINE CONCLUIDO!")
    print("="*60)
    print("\n Agora rode: python src/models/train_xgb.py")

if __name__ == "__main__":
    main()
