import sys
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import multiprocessing
from stable_baselines3.common.monitor import Monitor


# Garante que o Python encontre nossos módulos da pasta src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.database import DatabaseManager
from src.environment.trading_env import BitcoinTradingEnv
def make_env(df, rank, seed=0):
    def _init():
        env = BitcoinTradingEnv(df, initial_balance=10000.0)
        # O Monitor deve ser aplicado ANTES de qualquer outro wrapper
        env = Monitor(env) 
        env.reset(seed=seed + rank)
        return env
    return _init
def main():
    print("🧠 Iniciando o Pipeline de Treinamento de Inteligência Artificial...")

    # 1. Carregar os dados de treino do nosso Banco de Dados
    db = DatabaseManager('data/trading_data.db')
    df = db.load_data('btc_1h_features')
    
    if df is None or len(df) == 0:
        print("🚨 Erro: Sem dados para treinar. Execute o fetcher de dados primeiro.")
        return

    # Separar os dados: 80% para treino, 20% para validação (Walk-forward testing)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    
    print(f"📊 Dados divididos: {len(train_df)} candles para Treino.")

    # Pega o número de threads do seu PC (deixe 1 ou 2 livres para o SO não travar)
    num_cpu = multiprocessing.cpu_count() - 2
    if num_cpu < 1: num_cpu = 1
    
    print(f"⚡ Iniciando treinamento pesado utilizando {num_cpu} Threads simultâneas...")

    # Cria N ambientes isolados em processos separados
    env = SubprocVecEnv([make_env(train_df, i) for i in range(num_cpu)])
    
    # Mantém a normalização
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Construir o Modelo
    # Como agora temos muito mais dados chegando simultaneamente, aumentamos o batch_size
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        ent_coef=0.10,          # alto no início — forçar exploração de trades
        gamma=0.995,            # horizonte mais longo para capturar lucro do trade
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 128]),
        tensorboard_log="./tensorboard_logs/"
    )
    # 4. Treinamento
    # Timesteps = quantas "decisões" ela vai tomar. 
    # Em produção, você usará milhões (ex: 1_000_000). Para teste inicial, usaremos 50.000.
    TIMESTEPS = 3_000_000
    print(f"🚀 Iniciando treinamento por {TIMESTEPS} passos...")
    
    model.learn(total_timesteps=TIMESTEPS)

    # 5. Salvar os "Pesos" E AS ESTATÍSTICAS DE NORMALIZAÇÃO
    os.makedirs('data/models_weights', exist_ok=True)
    model.save("data/models_weights/ppo_btc_v1")
    env.save("data/models_weights/vec_normalize.pkl") # <--- Salva a escala matemática
    
    print(f"✅ Treinamento concluído! Modelo salvo")

if __name__ == "__main__":
    main()