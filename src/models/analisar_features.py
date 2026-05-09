import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.features import FeatureEngineer

def main():
    print("🧠 Lendo a Mente do Oráculo (Feature Importance)...")
    
    # 1. Carrega o modelo já treinado
    model_path = "data/models_weights/xgb_oraculo_btc.json"
    if not os.path.exists(model_path):
        print("🚨 Modelo não encontrado!")
        return
        
    model = XGBClassifier()
    model.load_model(model_path)
    
    # 2. Pega os nomes das colunas na exata ordem que a IA aprendeu
    features = FeatureEngineer.get_feature_list()
    
    # 3. Extrai os pesos de importância de dentro da rede
    importances = model.feature_importances_
    
    # 4. Cria uma tabela para visualizarmos
    df_importances = pd.DataFrame({
        'Indicador': features,
        'Importancia_Percentual': importances * 100
    })
    
    # Ordena do mais importante para o mais inútil
    df_importances = df_importances.sort_values(by='Importancia_Percentual', ascending=False)
    
    print("\n=================================================")
    print(" 🏆 TOP 20 INDICADORES MAIS IMPORTANTES PARA A IA")
    print("=================================================")
    
    # Mostra o Top 20
    for index, row in df_importances.head(20).iterrows():
        print(f"{row['Importancia_Percentual']:05.2f}% | {row['Indicador']}")
        
    print("\n=================================================")
    print(" 🗑️ OS 10 INDICADORES MAIS INÚTEIS (RUÍDO)")
    print("=================================================")
    
    # Mostra os 10 piores
    for index, row in df_importances.tail(10).iterrows():
        print(f"{row['Importancia_Percentual']:05.2f}% | {row['Indicador']}")

if __name__ == "__main__":
    main()