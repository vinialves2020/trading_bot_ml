import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.database import DatabaseManager
from src.data_pipeline.features import FeatureEngineer

def evaluate_threshold(probabilidades, y_test, threshold, profit_target, stop_loss, fee=0.002):
    decisoes = (probabilidades >= threshold).astype(int)
    
    if np.sum(decisoes) == 0:
        return {"threshold": threshold, "trades": 0}
        
    cm = confusion_matrix(y_test, decisoes)
    if cm.shape == (2, 2):
        acertos = cm[1][1]
        erros = cm[0][1]
    else:
        acertos = np.sum(y_test[decisoes == 1] == 1)
        erros = np.sum(y_test[decisoes == 1] == 0)
        
    total_trades = acertos + erros
    win_rate = acertos / total_trades if total_trades > 0 else 0
    
    # Net Reward / Risk (subtraindo taxas da Binance)
    net_reward = profit_target - fee
    net_risk = stop_loss + fee
    
    # Expectativa Matemática (EV)
    ev = (win_rate * net_reward) - ((1 - win_rate) * net_risk)
    
    return {
        "threshold": threshold,
        "trades": total_trades,
        "acertos": acertos,
        "erros": erros,
        "win_rate": win_rate,
        "ev_percent": ev * 100 # % por trade
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTC/USDT')
    parser.add_argument('--timeframe', type=str, default='15m')
    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.timeframe
    prefix = f"{symbol.split('/')[0].lower()}_{timeframe}"

    print(f"\n========================================================")
    print(f" Calibrando Confiança para: {symbol} - {timeframe} ")
    print(f"========================================================")

    db = DatabaseManager('data/trading_data.db')
    df = db.load_data(f'{prefix}_features')

    if df is None or len(df) == 0:
        print(" Erro: Sem dados. Execute pipeline.py primeiro.")
        return

    # Sincronizar com os mesmos targets do bot_executor / train_xgb
    if timeframe == '1h':
        pt, sl = 0.015, 0.0075
        horizon = 8
    else:
        pt, sl = 0.009, 0.0045
        horizon = 32

    print(f" Gerando Target -> TP: {pt*100:.2f}%, SL: {sl*100:.2f}%, Horizonte: {horizon} velas...")
    df = FeatureEngineer.create_target(df, horizon=horizon, profit_target=pt, stop_loss=sl)

    features = FeatureEngineer.get_feature_list()
    available_features = [f for f in features if f in df.columns]
    df_clean = df[available_features + ['target']].dropna()

    X = df_clean[available_features]
    y = (df_clean['target'] == 1).astype(int)

    # Testar na janela final de dados (últimos 20%)
    train_size = int(len(df_clean) * 0.8)
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    model_path = f"data/models_weights/xgb_oraculo_{prefix}.json"
    if not os.path.exists(model_path):
        print(f" Modelo {model_path} não encontrado! Rode train_xgb.py primeiro.")
        return

    model = XGBClassifier()
    model.load_model(model_path)

    probabilidades = model.predict_proba(X_test)[:, 1]

    thresholds = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.60]

    print("\n--------------------------------------------------------------------------------")
    print("Confiança | Win Rate | Trades  (Acerto/Erro) | EV (Líquido c/ Taxa) ")
    print("--------------------------------------------------------------------------------")

    melhor_t = None
    melhor_ev = -999

    for t in thresholds:
        res = evaluate_threshold(probabilidades, y_test, t, pt, sl)
        if res['trades'] > 0:
            ev_str = f"+{res['ev_percent']:.3f}%" if res['ev_percent'] > 0 else f"{res['ev_percent']:.3f}%"
            print(f"  {int(t*100)}%+    |  {res['win_rate']*100:.1f}%   |   {res['trades']:>4}  ({res['acertos']:>3} / {res['erros']:>3})  |   {ev_str}")
            
            # Guardamos o threshold com melhor EV (desde que tenha um volume mínimo de trades para ser significante)
            if res['ev_percent'] > melhor_ev and res['trades'] > 5:
                melhor_ev = res['ev_percent']
                melhor_t = t
        else:
            print(f"  {int(t*100)}%+    |    --    |      0                |   -- ")

    if melhor_t:
        print("\n========================================================")
        print(f" [MELHOR] Threshold Recomendado: {int(melhor_t*100)}% (Melhor EV: +{melhor_ev:.3f}% por trade)")
        print(f"========================================================\n")

if __name__ == "__main__":
    main()
