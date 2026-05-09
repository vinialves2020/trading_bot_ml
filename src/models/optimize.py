import sys
import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Garante que o Python encontre nossos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.database import DatabaseManager
from src.environment.trading_env import BitcoinTradingEnv

# Carrega os dados GLOBALMENTE uma vez só para economizar memória
db = DatabaseManager('data/trading_data.db')
df = db.load_data('btc_1h_features')
train_size = int(len(df) * 0.8)

# DICA PRO: Para o Optuna ser rápido, não usamos os 44.000 candles de uma vez.
# Usamos uma "fatia" dos últimos 10.000 candles de treino (representa o mercado recente)
train_df = df.iloc[train_size - 10000:train_size].copy()

def objective(trial):
    # 1. O Espaço de Busca
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    # Aumentei o mínimo do ent_coef para forçar ela a explorar mais no início
    ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1, log=True) 
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    
    # 2. Prepara o Ambiente com os "Óculos" (VecNormalize)
    env = DummyVecEnv([lambda: Monitor(BitcoinTradingEnv(train_df, initial_balance=10000.0, mode='aggressive'))])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Cria o Modelo
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        gamma=gamma,
        batch_size=batch_size,
        verbose=0
    )

    # 4. Treino Rápido (Aumentado para 100k passos para ela ter tempo de fazer trades)
    try:
        model.learn(total_timesteps=100000)
    except Exception as e:
        env.close()
        raise optuna.exceptions.TrialPruned()

    # 5. Avaliação (Desligamos o modo treino dos óculos para ver a realidade)
    env.training = False
    env.norm_reward = False 
    
    # Avalia por 3 episódios
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3)
    
    env.close()

    return mean_reward

def main():
    print("🤖 Iniciando AutoML Quantitativo (Optuna)...")
    
    # Cria um "Estudo" com o objetivo de MAXIMIZAR o lucro (mean_reward)
    study = optuna.create_study(direction="maximize")
    
    # Roda 30 tentativas. Com seu PC potente, isso deve levar uns 15-30 minutos.
    # Se quiser testar rápido primeiro, mude n_trials para 5.
    study.optimize(objective, n_trials=30)

    # RESULTADOS
    print("\n" + "="*45)
    print("🏆 ESTUDO CONCLUÍDO! OS MELHORES PARÂMETROS SÃO:")
    print("="*45)
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    print("\n💡 Próximo Passo: Copie estes parâmetros e cole no seu 'train.py'!")

if __name__ == "__main__":
    main()