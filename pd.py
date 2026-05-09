from src.data_pipeline.database import DatabaseManager
from src.data_pipeline.features import FeatureEngineer

db = DatabaseManager('data/trading_data.db')

# 1. Carrega os novos dados de 15 minutos
df = db.load_data('btc_15m_features')

# 2. Cria o Gabarito de Day Trade (0.4% em 8 candles de 15m = 2 horas)
df_ml = FeatureEngineer.create_target(df, horizon=8, profit_target=0.004)

# 3. Salva na NOVA tabela exclusiva para o ML de 15 minutos
# Lembre-se de passar primeiro o DataFrame, depois a string!
db.save_data(df_ml, 'btc_15m_ml_dataset')
print("💾 Dataset de Machine Learning salvo com sucesso!")