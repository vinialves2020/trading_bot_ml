import ccxt
import pandas as pd
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.database import DatabaseManager

def fetch_funding_history():
    print("🏦 Iniciando extração do Funding Rate (Mercado de Futuros)...")
    
    # Para o Funding Rate, TEMOS de usar o mercado de Futuros (USDM)
    exchange = ccxt.binanceusdm({
        'enableRateLimit': True,
    })
    
    # O símbolo em Futuros no CCXT moderno tem este formato
    symbol = 'BTC/USDT:USDT' 
    
    # Vamos descarregar desde o início de 2022 (como fez com as velas)
    since = exchange.parse8601("2022-01-01T00:00:00Z")
    all_funding = []
    
    while True:
        try:
            # Descarrega o histórico da taxa
            rates = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
            
            if not rates:
                break
                
            all_funding.extend(rates)
            since = rates[-1]['timestamp'] + 1
            
            last_date = pd.to_datetime(rates[-1]['timestamp'], unit='ms')
            print(f"Progresso Funding: Dados extraídos até {last_date}...")
            time.sleep(0.5)
            
        except Exception as e:
            print(f"🚨 Erro na API: {e}")
            time.sleep(5)

    # Converte para DataFrame
    df_funding = pd.DataFrame(all_funding)
    df_funding['timestamp'] = pd.to_datetime(df_funding['timestamp'], unit='ms')
    
    # O Funding Rate na Binance é cobrado de 8 em 8 horas. 
    # Precisamos preencher os espaços em branco para que cada vela de 15m tenha o valor da taxa daquele momento.
    df_funding.set_index('timestamp', inplace=True)
    df_funding = df_funding[['fundingRate']].astype(float)
    
    return df_funding

def main():
    db = DatabaseManager('data/trading_data.db')
    
    # 1. Carrega as velas antigas de 15m (OHLCV puro)
    df_ohlcv = db.load_data('btc_15m_features')
    
    # 2. Descarrega o histórico do Funding
    df_funding = fetch_funding_history()
    
    # 3. A MÁGICA: Junta a taxa às velas de 15 minutos!
    print("🔄 Sincronizando o Funding Rate com o gráfico de 15 Minutos...")
    # Usamos ffill() (Forward Fill) porque a taxa mantém-se igual durante 8 horas até ser atualizada
    df_merged = df_ohlcv.join(df_funding, how='left').ffill()
    
    # Limpa possíveis NaNs no início
    df_merged.dropna(subset=['fundingRate'], inplace=True)
    
    # 4. Guarda numa nova tabela Master
    db.save_data(df_merged, 'btc_15m_master')
    print("✅ Dados Alternativos fundidos com sucesso na tabela 'btc_15m_master'!")

if __name__ == "__main__":
    main()