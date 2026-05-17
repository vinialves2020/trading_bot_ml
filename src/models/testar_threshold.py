import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.database import DatabaseManager
from src.data_pipeline.features import FeatureEngineer

def main():
    print(" Calibrando a Confiana da Inteligncia Artificial...")

    # 1. Carrega os dados da tabela correta
    db = DatabaseManager('data/trading_data.db')
    df = db.load_data('btc_15m_features')

    if df is None or len(df) == 0:
        print(" Erro: Sem dados. Execute pipeline.py primeiro.")
        return

    # Recria target se nao existir
    if 'target' not in df.columns:
        df = FeatureEngineer.create_target(df, horizon=24, profit_target=0.004, stop_loss=0.002)

    features = FeatureEngineer.get_feature_list()
    available_features = [f for f in features if f in df.columns]
    df_clean = df[available_features + ['target']].dropna()

    X = df_clean[available_features]
    
    # Transforma o alvo em Binário: 1 (Vitória LONG) e 0 (Derrota ou Lateralização)
    y = (df_clean['target'] == 1).astype(int)

    train_size = int(len(df_clean) * 0.8)
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    # 2. Carrega o Modelo
    model_path = "data/models_weights/xgb_oraculo_btc.json"
    if not os.path.exists(model_path):
        print(" Modelo nao encontrado! Rode train_xgb.py primeiro.")
        return

    model = XGBClassifier()
    model.load_model(model_path)

    # 3. Pega as probabilidades
    probabilidades = model.predict_proba(X_test)[:, 1]

    # 4. Teste de Filtros (Buscando ~50% de Win Rate / Precisao)
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

    print("\n========================================================")
    print("  TESTE DE NIVEIS DE CONFIANCA (Buscando ~50% Win Rate) ")
    print("========================================================")
    print("Confiana | Precisao | Trades Acertados | Trades Errados")
    print("--------------------------------------------------------")

    for t in thresholds:
        decisoes = (probabilidades >= t).astype(int)

        if np.sum(decisoes) > 0:
            precisao = precision_score(y_test, decisoes, zero_division=0)
            cm = confusion_matrix(y_test, decisoes)
            acertos = cm[1][1] if cm.shape == (2, 2) else 0
            erros = cm[0][1] if cm.shape == (2, 2) else 0

            print(f"  {int(t*100)}%+    |   {precisao*100:.1f}%  |      {acertos}        |     {erros}")
        else:
            print(f"  {int(t*100)}%+    |  0 trades realizados (Confiana muito alta)")

if __name__ == "__main__":
    main()
