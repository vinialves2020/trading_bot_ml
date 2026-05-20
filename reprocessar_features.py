import pandas as pd
import sqlite3
from src.data_pipeline.database import DatabaseManager
from src.data_pipeline.features import FeatureEngineer

def main():
    print("🔄 A recalcular todas as features e o Target...")
    db = DatabaseManager('data/trading_data.db')
    
    # 1. Carrega os dados brutos CORRETOS
    df_antigo = db.load_data('btc_15m_raw')
    
    if df_antigo is None or df_antigo.empty:
        print("❌ Erro: Não foi possível carregar os dados brutos.")
        return

    # Arrumando o índice de tempo
    if 'timestamp' in df_antigo.columns:
        df_antigo['timestamp'] = pd.to_datetime(df_antigo['timestamp'])
        df_antigo.set_index('timestamp', inplace=True)
    
    # 2. Passa o DataFrame pelo motor de features
    df_novo = FeatureEngineer.apply_indicators(df_antigo)
    
    # 3. Cria o Gabarito (Target)
    df_ml = FeatureEngineer.create_target(df_novo, horizon=16, profit_target=0.006, stop_loss=0.003)
    df_ml = df_ml.dropna(subset=['target'])
    
    # --- A MÁGICA ENTRA AQUI ---
    # Deletamos a tabela velha com 58 colunas para dar espaço à nova com 59 colunas
    print(" Apagando formato antigo do banco de dados...")
    try:
        conn = sqlite3.connect('data/trading_data.db')
        conn.execute("DROP INDEX IF EXISTS ix_btc_15m_features_timestamp")
        conn.execute("DROP TABLE IF EXISTS btc_15m_features")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Aviso ao limpar tabela: {e}")
    # ----------------------------
    
    # 4. Salva a tabela limpa e estruturada
    db.save_data(df_ml, 'btc_15m_features', if_exists='replace')
    
    print("✅ Base de dados atualizada e salva com a coluna 'target'!")

if __name__ == "__main__":
    main()