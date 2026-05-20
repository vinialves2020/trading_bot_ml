import pandas as pd
import numpy as np
# Importações específicas da biblioteca 'ta'
from ta.trend import ema_indicator, sma_indicator, macd, macd_signal, macd_diff, adx
from ta.momentum import rsi, stochrsi_k
from ta.volatility import average_true_range, bollinger_hband, bollinger_lband, bollinger_mavg, bollinger_wband

class FeatureEngineer:
    @staticmethod
    def apply_indicators(df):
        print(" Calculando Indicadores Técnicos (Feature Engineering)...")
        df = df.copy()

        # 1. TENDÊNCIA (Renomeando para manter compatibilidade com o modelo original)
        df['EMA_9'] = ema_indicator(df['close'], window=9)
        df['EMA_21'] = ema_indicator(df['close'], window=21)
        df['SMA_200'] = sma_indicator(df['close'], window=200)
        # ADX (Average Directional Index) for choppiness filter
        df['ADX_14'] = adx(df['high'], df['low'], df['close'], window=14)

        # Sinal macro binário
        df['macro_trend'] = np.where(df['close'] > df['SMA_200'], 1.0, -1.0)
        df['dist_ema9']  = (df['close'] - df['EMA_9'])  / df['EMA_9']
        df['dist_ema21'] = (df['close'] - df['EMA_21']) / df['EMA_21']

        # 2. MOMENTUM DE PREÇO
        df['ret_1h']  = df['close'].pct_change(1)
        df['ret_4h']  = df['close'].pct_change(4)
        df['ret_12h'] = df['close'].pct_change(12)
        df['ret_24h'] = df['close'].pct_change(24)
        df['momentum_accel'] = df['ret_1h'] - df['ret_1h'].shift(1)

        # 3. POSIÇÃO NO RANGE
        rolling_high_24 = df['high'].rolling(24).max()
        rolling_low_24  = df['low'].rolling(24).min()
        df['price_position_24h'] = (df['close'] - rolling_low_24) / (rolling_high_24 - rolling_low_24 + 1e-9)

        rolling_high_4 = df['high'].rolling(4).max()
        rolling_low_4  = df['low'].rolling(4).min()
        df['price_position_4h'] = (df['close'] - rolling_low_4) / (rolling_high_4 - rolling_low_4 + 1e-9)

        # 4. VOLUME
        df['volume_ratio_24h'] = df['volume'] / (df['volume'].rolling(24).mean() + 1e-9)
        df['volume_ratio_4h']  = df['volume'] / (df['volume'].rolling(4).mean() + 1e-9)

        # 5. MOMENTUM CLÁSSICO
        df['RSI_14'] = rsi(df['close'], window=14)
        df['RSI_norm'] = (df['RSI_14'] - 50.0) / 50.0

        # MACD (Nomes compatíveis: MACD_12_26_9)
        df['MACD_12_26_9'] = macd(df['close'], window_slow=26, window_fast=12)
        df['MACDs_12_26_9'] = macd_signal(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACDh_12_26_9'] = macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)

        # 6. VOLATILIDADE
        df['ATRr_14'] = average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_pct'] = df['ATRr_14'] / df['close']

        # Bollinger Bands (Mapeando nomes do pandas_ta)
        df['BBL_20_2.0'] = bollinger_lband(df['close'], window=20, window_dev=2)
        
        # A média móvel central NÃO aceita desvio padrão (window_dev)
        df['BBM_20_2.0'] = bollinger_mavg(df['close'], window=20) 
        
        df['BBU_20_2.0'] = bollinger_hband(df['close'], window=20, window_dev=2)
        df['BBB_20_2.0_2.0'] = bollinger_wband(df['close'], window=20, window_dev=2)

        # 8. FEATURES DE DAY TRADE
        # VWAP Manual (Mais estável que via lib para dados intraday)
        vwap_cum_vol = df['volume'].cumsum()
        vwap_cum_pv = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
        df['VWAP_D'] = vwap_cum_pv / vwap_cum_vol
        df['dist_vwap'] = (df['close'] - df['VWAP_D']) / df['VWAP_D']

        # StochRSI
        df['STOCHRSIk_14_14_3_3'] = stochrsi_k(df['close'], window=14, smooth1=3, smooth2=3) * 100
        df['StochRSI_norm'] = (df['STOCHRSIk_14_14_3_3'] - 50.0) / 50.0 

        # 9. CONTEXTO MACRO
        df['SMA_96'] = sma_indicator(df['close'], window=96)
        df['EMA_16'] = ema_indicator(df['close'], window=16)
        df['macro_trend_daily'] = np.where(df['close'] > df['SMA_96'], 1.0, -1.0)
        df['macro_trend_4h'] = np.where(df['close'] > df['EMA_16'], 1.0, -1.0)

        # 10. FILTRO DE SESSÃO
        df['hour'] = df.index.hour
        df['is_us_session'] = df['hour'].isin([13, 14, 15, 16, 17, 18, 19, 20]).astype(float)
        df['atr_mean_daily'] = df['ATRr_14'].rolling(96).mean()
        df['volatility_regime'] = (df['ATRr_14'] / (df['atr_mean_daily'] + 1e-9))

        # 11. MICROESTRUTURA
        df['vol_sma'] = df['volume'].rolling(window=20).mean()
        df['vol_std'] = df['volume'].rolling(window=20).std()
        df['volume_z_score'] = (df['volume'] - df['vol_sma']) / (df['vol_std'] + 1e-9)

        candle_range = df['high'] - df['low'] + 1e-9
        df['body_size_ratio'] = abs(df['close'] - df['open']) / candle_range
        df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / candle_range
        df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / candle_range

        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['momentum_3'] = df['log_return'].rolling(3).sum()
        df['momentum_8'] = df['log_return'].rolling(8).sum()

        # 12. FUNDING RATE (Se existir)
        if 'fundingRate' in df.columns:
            df['funding_rate'] = df['fundingRate']
            df['funding_sma'] = df['funding_rate'].rolling(window=672).mean()
            df['funding_std'] = df['funding_rate'].rolling(window=672).std()
            df['funding_z_score'] = (df['funding_rate'] - df['funding_sma']) / (df['funding_std'] + 1e-9)

        df.dropna(inplace=True)
        print(f" Features calculadas. Colunas geradas: {len(df.columns)}")
        return df

    @staticmethod
    def create_target(df, horizon=16, profit_target=0.006, stop_loss=0.003):
        print(f" Gerando Target (Triple Barrier) com Alvo: {profit_target*100:.1f}%, Stop: {stop_loss*100:.1f}%, Horizonte: {horizon} velas...")
        df = df.copy()
        
        # Converte para arrays do numpy para processamento ultrarrápido
        import numpy as np
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # Array vazio para guardar o gabarito
        targets = np.full(len(df), np.nan)
        
        # Loop veloz pelo histórico (ignora as últimas velas do horizonte)
        for i in range(len(df) - horizon):
            entry_price = closes[i]
            tp_price = entry_price * (1 + profit_target)
            sl_price = entry_price * (1 - stop_loss)
            
            outcome = 0 # 0 = Timeout (Ficou lateral e não bateu nos alvos)
            
            # Olha para o futuro (horizon)
            for j in range(1, horizon + 1):
                idx = i + j
                
                # Regra de Ouro: O Stop Loss é prioridade. Se o pavio bater no Stop, aborta.
                if lows[idx] <= sl_price:
                    outcome = -1
                    break
                # Se o pavio bater no Take Profit sem bater no Stop antes:
                elif highs[idx] >= tp_price:
                    outcome = 1
                    break
                    
            targets[i] = outcome
            
        # Aplica o resultado de volta ao DataFrame
        df['target'] = targets
        
        return df

    @staticmethod
    def get_feature_list():
        # Retorna a lista exata que o bot espera encontrar
        return [
            'atr_pct', 'BBB_20_2.0_2.0', 'volatility_regime', 'hour', 'is_us_session',
            'funding_rate', 'funding_z_score', 'macro_trend', 'macro_trend_4h',
            'dist_ema21', 'dist_vwap', 'ret_1h', 'ret_4h', 'momentum_3', 'RSI_norm',
            'StochRSI_norm', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'volume_ratio_24h', 'price_position_4h', 'ADX_14'
        ]