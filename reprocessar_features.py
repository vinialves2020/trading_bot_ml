from src.data_pipeline.database import DatabaseManager
from src.data_pipeline.features import FeatureEngineer

def main():
    print("🔄 A recalcular todas as features com o novo Contexto Macro...")
    db = DatabaseManager('data/trading_data.db')
    
    # 1. Carregamos os dados que já temos
    df_antigo = db.load_data('btc_15m_master')
    
    # 2. Passamos o DataFrame pelo nosso novo motor de features
    df_novo = FeatureEngineer.apply_indicators(df_antigo)
    
    # 3. Recriamos o Gabarito (Target)
    df_ml = FeatureEngineer.create_target(df_novo, horizon=16, profit_target=0.009, stop_loss=0.003)
    
    # 4. Guardamos por cima do dataset de ML
    db.save_data(df_ml, 'btc_15m_ml_dataset')
    print("✅ Base de dados perfeitamente atualizada com as novas Features!")

if __name__ == "__main__":
    main()