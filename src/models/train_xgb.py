"""
Treino XGBoost - Versao Refatorada (finbert_training_prompt.md Parte 1)
Mudancas:
1. Walk-Forward Validation (substitui split estatico 80/20)
2. Modelo Dual: XGBoost (direcao) + LightGBM (magnitude)
3. Kelly Criterion para position sizing
4. Metricas: Sharpe, Sortino, MaxDD (em vez de acuracia)
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from xgboost import XGBClassifier
import lightgbm as lgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.database import DatabaseManager
from src.data_pipeline.features import FeatureEngineer

def walk_forward_split(df, n_splits=5, test_size=0.2):
    """
    Walk-Forward: treina em janelas deslizantes e testa no periodo seguinte.
    Retorna lista de (train_idx, test_idx).
    """
    n = len(df)
    step = int(n * test_size / n_splits)
    splits = []

    for i in range(n_splits):
        train_end = int(n * (0.2 + i * test_size / n_splits))
        test_start = train_end
        test_end = min(test_start + step, n)

        if test_start >= n:
            break

        splits.append((list(range(0, train_end)), list(range(test_start, test_end))))

    return splits

def train_direction_model(X_train, y_train, X_val=None, y_val=None):
    """Treina XGBoost para classificacao (direcao: alta/baixa)"""
    print("\n Treinando Modelo de Direcao (XGBoost)...")

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        scale_pos_weight=scale,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    return model

def train_magnitude_model(X_train, y_train,X_val=None, y_val=None):
    """
    Modelo Dual: LightGBM para prever magnitude do movimento (regressao).
    y_train = retorno percentual futuro em decimal.
    """
    print("\n Treinando Modelo de Magnitude (LightGBM)...")

    if hasattr(X_train, 'index') and hasattr(y_train, 'index'):
        y_train.index = X_train.index
        
    mask = y_train > 0
    
    # Aplica o filtro de forma segura
    X = X_train[mask]
    y = y_train[mask]

    if len(X) == 0:
        print(" Sem dados para treinar magnitude")
        return None

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        verbose=-1
    )
    model.fit(X, y)
    return model

def calculate_kelly_fraction(win_rate, avg_win, avg_loss):
    """
    Kelly Criterion para position sizing.
    f* = (p*b - q) / b, onde b = avg_win/avg_loss, p = win_rate, q = 1-p
    """
    if avg_loss == 0 or win_rate == 0:
        return 0.02

    b = abs(avg_win / avg_loss)
    p = win_rate
    q = 1 - p

    kelly = (p * b - q) / b
    kelly = max(0.0, min(kelly, 0.25))

    return kelly

def evaluate_with_sharpe(y_true, y_pred, returns=None):
    """
    Metricas institucionais: Sharpe, Sortino, MaxDD.
    Substitui acuracia simples.
    """
    if returns is None:
        return None

    returns = np.array(returns)
    if len(returns) == 0:
        return None

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(24192) if std_ret != 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std()
    sortino = (mean_ret / downside_std) * np.sqrt(24192) if downside_std != 0 else 0.0

    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_dd = abs(drawdown.min())

    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'total_trades': len(returns)
    }

def main():
    print("=" * 60)
    print(" REFATORACAO DO TREINO (finbert_training_prompt.md)")
    print("=" * 60)

    db = DatabaseManager('data/trading_data.db')
    df = db.load_data('btc_15m_features')

    if df is None or len(df) == 0:
        print(" Erro: Sem dados. Execute o pipeline.py primeiro.")
        return

    print(f" Dados carregados: {len(df)} candles")
    print(f" Periodo: {df.index[0]} a {df.index[-1]}")

    # Aumentar horizonte para 6h (24 candles de 15m) - mais tempo para o preço se mover
    df = FeatureEngineer.create_target(df, horizon=32, profit_target=0.009, stop_loss=0.003)

    print(f"\n Walk-Forward Validation (5 splits)...")
    features_list = FeatureEngineer.get_feature_list()

    features = FeatureEngineer.get_feature_list()
    available_features = [f for f in features if f in df.columns]
    df_clean = df[available_features + ['target']].dropna()

    X = df_clean[available_features]
    
    # Ajuste binário para o cálculo de precisão
    y = (df_clean['target'] == 1).astype(int)

    df['future_return'] = df['close'].pct_change(periods=16).shift(-16)
    y_magnitude = df['future_return']

    splits = walk_forward_split(df, n_splits=5, test_size=0.2)
    print(f" Splits gerados: {len(splits)}")

    direction_models = []
    magnitude_models = []
    metrics_history = []

    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"\n--- Split {i+1}/{len(splits)} ---")
        print(f" Treino: {train_idx[0]} a {train_idx[-1]} ({len(train_idx)} amostras)")
        print(f" Teste: {test_idx[0]} a {test_idx[-1]} ({len(test_idx)} amostras)")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_dir = train_direction_model(X_train, y_train)
        direction_models.append(model_dir)

        y_mag_train = y_magnitude.iloc[train_idx]
        model_mag = train_magnitude_model(X_train, y_mag_train)
        magnitude_models.append(model_mag)

        prob = model_dir.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        entry_prices = df['close'].iloc[test_idx].values
        future_prices = df['close'].iloc[test_idx].shift(-16).values

        returns = []
        for j in range(len(test_idx) - 16):
            if pred[j] == 1 and future_prices[j] > 0:
                ret = (future_prices[j] - entry_prices[j]) / entry_prices[j]
                returns.append(ret)

        metrics = evaluate_with_sharpe(y_test.iloc[:len(returns)], pred[:len(returns)], returns)
        if metrics:
            print(f" Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f}")
            print(f" Max DD: {metrics['max_drawdown']*100:.2f}% | Trades: {metrics['total_trades']}")
            metrics_history.append(metrics)

    print(f"\n Treinando Modelo Final (Deploy) com hiperparametros otimizados...")

    X_final = X.fillna(0)
    y_final = y.fillna(0)

    # Melhores parametros encontrados pelo Optuna (Trial 14)
    best_params = {
        'n_estimators': 217,
        'max_depth': 6,
        'learning_rate': 0.0215,
        'subsample': 0.6869,
        'colsample_bytree': 0.5041,
        'scale_pos_weight': 5.7684,
        'random_state': 42,
        'n_jobs': -1
    }

    final_direction = XGBClassifier(**best_params)
    final_direction.fit(X_final, y_final)

    y_magnitude_aligned = y_magnitude.loc[X_final.index]
    
    final_magnitude = train_magnitude_model(X_final, y_magnitude_aligned.fillna(0))

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    models_dir = os.path.join(base_path, "data", "models_weights")
    os.makedirs(models_dir, exist_ok=True)

    direction_path = os.path.join(models_dir, "xgb_oraculo_btc.json")
    final_direction.save_model(direction_path)
    print(f" Modelo Direcao salvo: {direction_path}")

    if final_magnitude:
        magnitude_path = os.path.join(models_dir, "lgbm_magnitude_btc.txt")
        final_magnitude.booster_.save_model(magnitude_path)
        print(f" Modelo Magnitude salvo: {magnitude_path}")

    win_rate = df['target'].mean()
    avg_win = 0.004
    avg_loss = 0.002

    kelly = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
    print(f"\n Kelly Fracionario sugerido: {kelly*100:.2f}% do capital por trade")
    print(f"   (Atualmente usando 2% fixo - considere ajustar para {kelly*100:.2f}%)")

    if metrics_history:
        avg_sharpe = np.mean([m['sharpe'] for m in metrics_history])
        avg_sortino = np.mean([m['sortino'] for m in metrics_history])
        avg_dd = np.mean([m['max_drawdown'] for m in metrics_history])

        metrics_summary = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'XGBoost_Direction + LightGBM_Magnitude',
            'walk_forward_splits': len(splits),
            'avg_sharpe_ratio': float(avg_sharpe),
            'avg_sortino_ratio': float(avg_sortino),
            'avg_max_drawdown': float(avg_dd),
            'kelly_fraction': float(kelly),
            'features_count': len(available_features)
        }

        import json
        metrics_path = os.path.join(base_path, "data", "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        print(f"\n Metricas salvas em: {metrics_path}")
        print(f"   Sharpe Medio: {avg_sharpe:.2f}")
        print(f"   Sortino Medio: {avg_sortino:.2f}")
        print(f"   MaxDD Medio: {avg_dd*100:.2f}%")

    print("\n" + "="*60)
    print(" TREINO REFATORADO CONCLUIDO!")
    print("="*60)

if __name__ == "__main__":
    main()
