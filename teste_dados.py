# teste_dados.py
from src.data_pipeline.fetcher import BinanceDataFetcher
from src.data_pipeline.features import FeatureEngineer
from src.data_pipeline.database import DatabaseManager

def main():
    # 1. Inicializa os módulos
    fetcher = BinanceDataFetcher(symbol='BTC/USDT', timeframe='15m')
    db_manager = DatabaseManager()
    
    # Nome da tabela onde vamos salvar
    table_name = "btc_15m_features"

    # 2. Baixa e processa os dados
    df_bruto = fetcher.fetch_deep_history() # Vamos pegar 1000 candles agora
    df_processado = FeatureEngineer.apply_indicators(df_bruto)

    # 3. Salva no banco de dados (SQLite)
    # Usamos 'replace' no teste para sempre recriar a tabela. Em produção usaremos 'append'.
    db_manager.save_data(df_processado, table_name=table_name, if_exists='replace')

    print("\n--- Fechando conexões e simulando um novo dia ---\n")

    # 4. Carrega os dados do banco de dados (Sem bater na API da Binance!)
    df_carregado = db_manager.load_data(table_name)

    # 5. Validação final
    print("\nÚltimas 3 linhas carregadas do Banco de Dados:")
    print(df_carregado[['close', 'EMA_9', 'RSI_14', 'ATRr_14']].tail(3))

if __name__ == "__main__":
    main()