import sys
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # <--- NOVO AQUI
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.database import DatabaseManager
from src.environment.trading_env import BitcoinTradingEnv

def run_backtest():
    print("🔍 Iniciando Backtesting em Dados Não Vistos (Out-of-Sample)...")

    db = DatabaseManager('data/trading_data.db')
    df = db.load_data('btc_1h_features')

    if df is None or len(df) == 0:
        return

    # Separar os 20% finais
    train_size = int(len(df) * 0.8)
    test_df = df.iloc[train_size:].copy()
    
    print(f"📈 Período de Teste: {test_df.index[0]} até {test_df.index[-1]} ({len(test_df)} candles)")

    INITIAL_BALANCE = 10000.0
    
    # 1. Cria o ambiente base
    env = DummyVecEnv([lambda: BitcoinTradingEnv(test_df, initial_balance=INITIAL_BALANCE)])
    
    # 2. CARREGA O NORMALIZADOR SALVO NO TREINO (Crucial!)
    norm_path = "data/models_weights/vec_normalize.pkl"
    try:
        env = VecNormalize.load(norm_path, env)
        # Desliga a atualização da média/variância durante o teste para não "roubar" vendo o futuro
        env.training = False 
        # Desliga a normalização da recompensa, pois queremos ver dólares reais
        env.norm_reward = False 
    except Exception as e:
        print(f"🚨 Erro ao carregar normalizador. Arquivo {norm_path} existe? Erro: {e}")
        return

    # 3. Carrega o Modelo
    model_path = "data/models_weights/ppo_btc_v1"
    try:
        model = PPO.load(model_path)
        print("✅ Cérebro da IA carregado com sucesso!")
    except Exception as e:
        return

    obs = env.reset()
    done = False
    net_worth_history = []
    
    action_counts = {0: 0, 1: 0, 2: 0}
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        current_info = info[0]
        net_worth_history.append(current_info['net_worth'])
        action_counts[int(current_info['action'])] += 1
        done = done[0] 

    final_balance = net_worth_history[-1]
    roi = ((final_balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    
    print("\n" + "="*45)
    print("📊 RESULTADOS DO BACKTEST INSTITUCIONAL")
    print("="*45)
    print(f"Saldo Inicial:   USDT {INITIAL_BALANCE:.2f}")
    print(f"Saldo Final:     USDT {final_balance:.2f}")
    print(f"Retorno (ROI):   {roi:.2f}%")
    print("="*45)

    print(f"\n📊 Distribuição de ações:")
    total = sum(action_counts.values())
    for a, count in action_counts.items():
        label = ['Neutro', 'Long', 'Short'][a]
        print(f"  {label}: {count} ({count/total*100:.1f}%)")

    # Gráfico (Apenas se não faliu)
    if len(net_worth_history) > 0:
        fig = go.Figure()
        fig.add_hline(y=INITIAL_BALANCE, line_dash="dash", line_color="gray", annotation_text="Break-even")
        fig.add_trace(go.Scatter(
            x=test_df.index, y=net_worth_history, mode='lines',
            name='Patrimônio do Bot',
            line=dict(color='#00ff00' if roi >= 0 else '#ff0000', width=2)
        ))
        fig.update_layout(
            title='Curva de Patrimônio (Equity Curve) - Out of Sample',
            yaxis_title='Saldo Total (USDT)', template='plotly_dark'
        )
        fig.write_html("backtest_results.html")
        print("\n📈 Gráfico interativo gerado: backtest_results.html")

if __name__ == "__main__":
    run_backtest()