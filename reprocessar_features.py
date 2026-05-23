import pandas as pd
import sqlite3
import argparse
from src.data_pipeline.database import DatabaseManager
from src.data_pipeline.features import FeatureEngineer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Ex: BTC/USDT, ETH/USDT, SOL/USDT')
    parser.add_argument('--timeframe', type=str, default='15m', help='Ex: 15m, 1h')
    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.timeframe
    prefix = f"{symbol.split('/')[0].lower()}_{timeframe}"
    
    print(f"🔄 A recalcular todas as features e o Target para {symbol} ({timeframe})...")
    db = DatabaseManager('data/trading_data.db')
    
    # 1. Carrega os dados brutos CORRETOS
    df_antigo = db.load_data(f'{prefix}_raw')
    
    if df_antigo is None or df_antigo.empty:
        print(f"❌ Erro: Não foi possível carregar os dados brutos para {prefix}_raw.")
        return

    # Arrumando o índice de tempo
    if 'timestamp' in df_antigo.columns:
        df_antigo['timestamp'] = pd.to_datetime(df_antigo['timestamp'])
        df_antigo.set_index('timestamp', inplace=True)
    
    # 2. Passa o DataFrame pelo motor de features
    df_novo = FeatureEngineer.apply_indicators(df_antigo)
    
    # 3. Cria o Gabarito (Target)
    if timeframe == '1h':
        df_ml = FeatureEngineer.create_target(df_novo, horizon=8, profit_target=0.015, stop_loss=0.0075)
    else:
        df_ml = FeatureEngineer.create_target(df_novo, horizon=16, profit_target=0.006, stop_loss=0.003)
        
    df_ml = df_ml.dropna(subset=['target_long', 'target_short'])
    
    # --- A MÁGICA ENTRA AQUI ---
    print(f" Apagando formato antigo do banco de dados para {prefix}_features...")
    try:
        conn = sqlite3.connect('data/trading_data.db')
        conn.execute(f"DROP INDEX IF EXISTS ix_{prefix}_features_timestamp")
        conn.execute(f"DROP TABLE IF EXISTS {prefix}_features")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Aviso ao limpar tabela: {e}")
    # ----------------------------
    
    # 4. Salva a tabela limpa e estruturada
    db.save_data(df_ml, f'{prefix}_features', if_exists='replace')
    
    print(f"✅ Base de dados {prefix}_features atualizada e salva com as novas colunas Target!")

if __name__ == "__main__":
    main()