import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, precision_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.database import DatabaseManager
from src.data_pipeline.features import FeatureEngineer

def objective(trial):
    # 1. Carrega os dados da tabela correta
    db = DatabaseManager('data/trading_data.db')
    df = db.load_data('btc_15m_features')

    if df is None or len(df) == 0:
        raise ValueError("Sem dados. Execute pipeline.py primeiro.")

    # Recria target se nao existir
    if 'target' not in df.columns:
        df = FeatureEngineer.create_target(df, horizon=24, profit_target=0.004, stop_loss=0.002)

    features = FeatureEngineer.get_feature_list()
    available_features = [f for f in features if f in df.columns]
    df_clean = df[available_features + ['target']].dropna()

    X = df_clean[available_features]
    y = df_clean['target']

    train_size = int(len(df_clean) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # 2. O Optuna escolhe aleatoriamente os valores dentro destes limites
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # O segredo est aqui: vamos deixar ele testar pesos menores para ser mais exigente
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 6.0), 
        'random_state': 42,
        'n_jobs': -1
    }

    # 3. Treina o crebro temporrio
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)

    # 4. Avalia a qualidade (PR-AUC  a melhor mtrica para dados desbalanceados)
    probabilidades = model.predict_proba(X_test)[:, 1]
    
    # Queremos que ele maximize a rea sob a curva de preciso
    score = average_precision_score(y_test, probabilidades)
    
    return score

def main():
    print(" Iniciando a Busca pela Arquitetura Perfeita (Optuna)...")
    
    # Pode precisar instalar o optuna se ainda no o tiver: pip install optuna
    study = optuna.create_study(direction='maximize')
    
    # N_trials = 30. Vai testar 30 configuraes diferentes.
    # Como o XGBoost  rpido, deve demorar poucos minutos.
    study.optimize(objective, n_trials=30)
    
    print("\n=============================================")
    print(" OS MELHORES PARMETROS ENCONTRADOS")
    print("=============================================")
    print(study.best_params)

if __name__ == "__main__":
    main()