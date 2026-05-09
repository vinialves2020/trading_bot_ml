import pandas as pd
import pandas_ta as ta
import numpy as np


class FeatureEngineer:
    @staticmethod
    def apply_indicators(df):
        print(" Calculando Indicadores Tcnicos (Feature Engineering)...")
        df = df.copy()

        #  1. TENDNCIA 
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.sma(length=200, append=True)

        # Sinal macro binrio: +1 acima da SMA200, -1 abaixo
        df['macro_trend'] = np.where(df['close'] > df['SMA_200'], 1.0, -1.0)

        # Distncia percentual da EMA9 e EMA21 (informa o "estado" da tendncia)
        df['dist_ema9']  = (df['close'] - df['EMA_9'])  / df['EMA_9']
        df['dist_ema21'] = (df['close'] - df['EMA_21']) / df['EMA_21']

        #  2. MOMENTUM DE PREO 
        # Retornos em mltiplos horizontes  sinal mais direto e no-lagging
        df['ret_1h']  = df['close'].pct_change(1)
        df['ret_4h']  = df['close'].pct_change(4)
        df['ret_12h'] = df['close'].pct_change(12)
        df['ret_24h'] = df['close'].pct_change(24)

        # Acelerao: derivada do momentum (2 diferena)
        df['momentum_accel'] = df['ret_1h'] - df['ret_1h'].shift(1)

        #  3. POSIO NO RANGE 
        # Onde o preo est dentro do range das ltimas 24h (0 = mnima, 1 = mxima)
        rolling_high_24 = df['high'].rolling(24).max()
        rolling_low_24  = df['low'].rolling(24).min()
        df['price_position_24h'] = (
            (df['close'] - rolling_low_24) /
            (rolling_high_24 - rolling_low_24 + 1e-9)
        )

        # Range das ltimas 4h (curto prazo)
        rolling_high_4 = df['high'].rolling(4).max()
        rolling_low_4  = df['low'].rolling(4).min()
        df['price_position_4h'] = (
            (df['close'] - rolling_low_4) /
            (rolling_high_4 - rolling_low_4 + 1e-9)
        )

        #  4. VOLUME 
        # Volume relativo: spike > 1.5 indica movimentos com "convico"
        df['volume_ratio_24h'] = df['volume'] / (df['volume'].rolling(24).mean() + 1e-9)
        df['volume_ratio_4h']  = df['volume'] / (df['volume'].rolling(4).mean() + 1e-9)

        #  5. MOMENTUM CLSSICO 
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        # RSI normalizado para [-1, 1] (rede neural processa melhor)
        df['RSI_norm'] = (df['RSI_14'] - 50.0) / 50.0

        #  6. VOLATILIDADE 
        df.ta.atr(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)

        # ATR normalizado pelo preo (volatilidade relativa, no absoluta)
        df['atr_pct'] = df['ATRr_14'] / df['close']

        #  8. FEATURES DE DAY TRADE (Intraday) 
        # VWAP (Preo Mdio Ponderado por Volume)
        df.ta.vwap(append=True)
        # Calcula a distncia do preo para a VWAP (O Segredo Institucional)
        df['dist_vwap'] = (df['close'] - df['VWAP_D']) / df['VWAP_D']

        # StochRSI (Reao rpida para pullbacks em 15m)
        df.ta.stochrsi(length=14, append=True)
        # O pandas_ta gera STOCHRSIk e STOCHRSId. Vamos usar o K.
        # Normalizando para [-1, 1]
        df['StochRSI_norm'] = (df['STOCHRSIk_14_14_3_3'] - 50.0) / 50.0 
        #  9. O CONTEXTO MACRO (Multi-Timeframe em 15m) 
        # Simulando o grfico de 1 Dia (96 velas) e 4 Horas (16 velas)
        df.ta.sma(length=96, append=True) # Tendncia Diria
        df.ta.ema(length=16, append=True) # Tendncia de 4 Horas

        # Sinal Macro: 1 se o preo estiver acima (Alta), -1 se abaixo (Baixa)
        df['macro_trend_daily'] = np.where(df['close'] > df['SMA_96'], 1.0, -1.0)
        df['macro_trend_4h'] = np.where(df['close'] > df['EMA_16'], 1.0, -1.0)

        #  10. O FILTRO DE SESSO E VOLATILIDADE (O Relgio da IA) 
        # Extrai a hora do dia (0 a 23)
        df['hour'] = df.index.hour
        
        # Cria uma flag para o horrio de pico institucional (Wall Street e Londres cruzados)
        # Aproximadamente das 13h s 20h (UTC)
        df['is_us_session'] = df['hour'].isin([13, 14, 15, 16, 17, 18, 19, 20]).astype(float)
        
        # Volatilidade Relativa: O mercado est agitado agora comparado com ontem?
        df['atr_mean_daily'] = df['ATRr_14'].rolling(96).mean()
        df['volatility_regime'] = (df['ATRr_14'] / (df['atr_mean_daily'] + 1e-9))


        #  11. MICROESTRUTURA E ANOMALIAS (O Segredo Institucional) 
        
        # A. Anomalia de Volume (Z-Score)
        # Calcula se o volume atual  uma aberrao estatstica em relao s ltimas 20 velas
        df['vol_sma'] = df['volume'].rolling(window=20).mean()
        df['vol_std'] = df['volume'].rolling(window=20).std()
        df['volume_z_score'] = (df['volume'] - df['vol_sma']) / (df['vol_std'] + 1e-9)

        # B. Anatomia da Vela (Fora contra Rejeio)
        # Tamanho do corpo em relao ao tamanho total da vela (0 a 1)
        candle_range = df['high'] - df['low'] + 1e-9
        df['body_size_ratio'] = abs(df['close'] - df['open']) / candle_range
        
        # Presso Vendedora Oculta (Tamanho do pavio superior)
        df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / candle_range
        # Presso Compradora Oculta (Tamanho do pavio inferior)
        df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / candle_range

        # C. Acelerao Pura (Log Returns Acumulados)
        # Velocidade do preo sem o atraso das mdias mveis
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['momentum_3'] = df['log_return'].rolling(3).sum() # Momentum de 45 minutos
        df['momentum_8'] = df['log_return'].rolling(8).sum() # Momentum de 2 horas

        #  12. DADOS ALTERNATIVOS (Institucionais) 
        if 'fundingRate' in df.columns:
            # 1. A Taxa Pura
            df['funding_rate'] = df['fundingRate']
            
            # 2. O Z-Score do Funding (Isto  ouro puro!)
            # Calcula se a ganncia/medo atual  extrema em relao  ltima semana (672 velas de 15m)
            df['funding_sma'] = df['funding_rate'].rolling(window=672).mean()
            df['funding_std'] = df['funding_rate'].rolling(window=672).std()
            df['funding_z_score'] = (df['funding_rate'] - df['funding_sma']) / (df['funding_std'] + 1e-9)
        # Limpeza final
        df.dropna(inplace=True)

        print(f" Features calculadas. Colunas geradas: {len(df.columns)}")
        print(f"   Total de linhas prontas para treino: {len(df)}")
        return df
    @staticmethod
    def create_target(df, horizon=16, profit_target=0.004, stop_loss=0.002):
        """
        Target Institucional: Triple Barrier Method
        Verifica se o preo atinge o Take Profit ANTES do Stop Loss no horizonte de tempo.
        ALINHADO COM O BOT: horizon=16 (4h), TP=0.4%, SL=0.2%
        """
        print(f" Criando Target (Triple Barrier): TP +{profit_target*100}% | SL -{stop_loss*100}% | Tempo Mx: {horizon} velas")
        df = df.copy()
        
        # Array para guardar os resultados
        targets = np.zeros(len(df))
        
        # Extrai os valores para numpy arrays (para processamento muito mais rpido)
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Varre os dados (exceto as ltimas velas que no tm futuro suficiente)
        for i in range(len(df) - horizon):
            entry_price = closes[i]
            tp_price = entry_price * (1 + profit_target)
            sl_price = entry_price * (1 - stop_loss)
            
            # Olha para a janela do futuro
            window_highs = highs[i+1 : i+1+horizon]
            window_lows = lows[i+1 : i+1+horizon]
            
            # Verifica qual barreira  atingida primeiro
            for j in range(horizon):
                # Bateu no Stop Loss primeiro? Fim de jogo (Target = 0)
                if window_lows[j] <= sl_price:
                    break
                
                # Bateu no Take Profit sem bater no Stop Loss? Vitria (Target = 1)
                if window_highs[j] >= tp_price:
                    targets[i] = 1
                    break
        
        df['target'] = targets
        
        # Remove as linhas finais que no puderam ser calculadas
        df = df.iloc[:-horizon]
        
        alvos_atingidos = df['target'].sum()
        total = len(df)
        print(f" Total de amostras: {total}")
        print(f" Operaes Lucrativas (TP antes do SL): {alvos_atingidos} ({alvos_atingidos/total*100:.2f}%)")
        
        return df

    @staticmethod
    def get_feature_list():
        """
        Lista EXATA das features que o ambiente vai usar.
        Mantenha sincronizado com trading_env.py  self.features_seguras
        """
        return [
            # 1. Volatilidade (Os Reis da Preciso)
            'atr_pct',
            'BBB_20_2.0_2.0',
            'volatility_regime',

            # 2. Timing e Sesso (Onde a liquidez est)
            'hour',
            'is_us_session',

            # 3. Institucional (O Medo e a Ganncia)
            'funding_rate',
            'funding_z_score',

            # 4. Tendncia e Contexto (Multi-Timeframe)
            'macro_trend',        # SMA 200
            'macro_trend_4h',     # EMA 16
            'dist_ema21',
            'dist_vwap',

            # 5. Momentum Curto (Acelerao Real)
            'ret_1h',
            'ret_4h',
            'momentum_3',
            'RSI_norm',
            'StochRSI_norm',

            # 6. Fora do Movimento (MACD)
            'MACD_12_26_9',
            'MACDh_12_26_9',
            'MACDs_12_26_9',

            # 7. Volume e Posio Relativa
            'volume_ratio_24h',
            'price_position_4h'
        ]
       