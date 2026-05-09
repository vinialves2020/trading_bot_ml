import ccxt
import pandas as pd
import time

class BinanceDataFetcher:
    """
    Classe responsvel por extrair dados OHLCV da Binance.
    """
    def __init__(self, symbol='BTC/USDT', timeframe='15m'):
        # Inicializa a conexo com a Binance.
        # enableRateLimit  crucial para no ser banido pela API.
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        self.exchange_futures = ccxt.binanceusdm({
            'enableRateLimit': True,
        })
        self.symbol = symbol
        self.timeframe = timeframe

    def fetch_deep_history(self, start_date_str="2022-01-01 00:00:00"):
        """
        Faz requisies em loop paginadas para baixar ANOS de histrico.
        """
        print(f"[FETCH] Iniciando extracao profunda desde {start_date_str} ({self.timeframe})...")
        
        # Converte a data string para timestamp Unix (milissegundos) exigido pela Binance
        since = self.exchange.parse8601(start_date_str.replace(" ", "T") + "Z")
        all_candles = []
        
        while True:
            try:
                # O parmetro 'since'  a chave para a paginao
                candles = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=1000)
                
                if not candles:
                    break # Fim dos dados disponveis
                    
                all_candles.extend(candles)
                
                # O prximo loop deve comear 1 milissegundo aps o ltimo candle baixado
                since = candles[-1][0] + 1
                
                # Pega a data legvel do ltimo candle do lote para mostrar no print
                last_date = pd.to_datetime(candles[-1][0], unit='ms')
                print(f"Progresso: Dados extrados at {last_date}...")
                
                # Respeita a API (Rate Limit extra de segurana)
                time.sleep(0.5) 
                
            except Exception as e:
                print(f" Erro durante paginao: {e}")
                time.sleep(5) # Se der erro de rate limit, espera 5 segundos e tenta de novo

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Remove duplicatas caso haja alguma sobreposio nos loops
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)

        # Funding rate constante para treino (bot usa real-time em execucao)
        df['fundingRate'] = 0.0001

        print(f"[OK] Extracao Concluida! Total de candles: {len(df)}")
        return df