import time
import ccxt
import pandas as pd
import numpy as np
import os
import sys
import json
import gc
from datetime import datetime, timezone, timedelta

# 1. Ajuste de Caminho
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.database import DatabaseManager
from src.data_pipeline.features import FeatureEngineer
from xgboost import XGBClassifier
import lightgbm as lgb

class TradingBot:
    def __init__(self, symbol='BTC/USDT', threshold=0.60, paper_trading=True):
        self.symbol = symbol
        self.threshold = threshold
        self.paper_trading = paper_trading
        self.db_manager = DatabaseManager('data/trading_data.db')

        # Conexões Binance Futures
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # USDT-margined futures
            }
        })

        self.exchange_futures = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # FinBERT Sentiment (carregamento tardio)
        self.finbert_available = False
        try:
            from src.models.finbert_sentiment import analisar_sentimento_btc
            self._get_sentiment = analisar_sentimento_btc
            self.finbert_available = True
            print(" FinBERT disponvel para anlise de sentimento")
        except ImportError:
            print(" Aviso: FinBERT no disponvel. Instale: pip install transformers torch")

        # Carregar Modelos (Dual: Direo + Magnitude)
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(base_path, "data", "models_weights")

        # 1. Modelo de Direo (XGBoost)
        model_path = os.path.join(model_dir, "xgb_oraculo_btc.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f" Modelo nao encontrado em: {model_path}")
        self.model = XGBClassifier()
        self.model.load_model(model_path)

        # 2. Modelo de Magnitude (LightGBM) - Parte 1 finbert_training_prompt.md
        magnitude_path = os.path.join(model_dir, "lgbm_magnitude_btc.txt")
        self.magnitude_available = False
        if os.path.exists(magnitude_path):
            try:
                self.model_magnitude = lgb.Booster(model_file=magnitude_path)
                self.magnitude_available = True
                print(" Modelo de Magnitude (LightGBM) carregado")
            except Exception as e:
                print(f" Aviso: Magnitude nao carregada: {e}")

        self.features_list = FeatureEngineer.get_feature_list()

        # Kelly Criterion (Parte 1 finbert_training_prompt.md)
        # Carregar mtricas de treino se existirem
        self.kelly_fraction = 0.02  # Default 2%
        metrics_path = os.path.join(base_path, "data", "training_metrics.json")
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)
                self.kelly_fraction = metrics.get('kelly_fraction') or 0.02
                print(f" Kelly Criterion: {self.kelly_fraction*100:.2f}% (do treino)")

        # Parametros da estrategia (ALINHADOS COM O TREINO)
        self.tp_pct = 0.012   # 1.2%
        self.sl_pct = 0.002   # 0.2%
        self.break_even_trigger_pct = 0.006 # 0.6% de lucro ativa a proteção
        self.break_even_target_pct = 0.002 # Move o stop para o ponto de entrada + 0.2% quando o lucro atingir 0.6%
        self.fee_rate = 0.001 # 0.1% Taxa da Binance (Maker/Taker)
        self.max_risk_per_trade = self.kelly_fraction  # Usa Kelly dinamico
        self.max_daily_drawdown = 0.05   # -5% suspende

        # Estado do bot
        self.daily_start_balance = None
        self.open_order = None
        self.paper_balance = 100.0  # Saldo simulado R$ 10.000
        self.paper_start_balance = 100.0
        self.trade_count = 0
        # Order tracking for limit order fallback
        self.order_timestamp = None
        self.order_id = None
        # Limit order tracking
        self.limit_order_price = None
        self.limit_order_side = None
        self.limit_order_timestamp = None

        # JSONL Trade Journaling (prompt.md)
        self.journal_file = os.path.join(base_path, "data", "trade_journal.jsonl")

        self._init_log_table()

    def _init_log_table(self):
        # Remove tabela antiga se existir (para corrigir schema)
        self.db_manager.execute_query("DROP TABLE IF EXISTS trade_history")
        query = """
        CREATE TABLE trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            side TEXT,
            entry_price REAL,
            take_profit REAL,
            stop_loss REAL,
            confidence REAL,
            position_size_usdt REAL,
            status TEXT DEFAULT 'OPEN'
        )
        """
        self.db_manager.execute_query(query)

    def _log_to_journal(self, trade_data):
        """Trade Journaling em JSONL conforme prompt.md linha 82-88"""
        with open(self.journal_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trade_data, default=str) + '\n')

    def _get_realtime_data(self):
        """Busca 1000 velas de 15m da Binance + Funding Rate REAL"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='15m', limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Funding Rate REAL
            try:
                funding = self.exchange_futures.fetch_funding_rate_history(
                    symbol='BTC/USDT:USDT', limit=1
                )
                current_funding = funding[0]['fundingRate'] if funding else 0.0001
            except Exception as e:
                print(f" Erro ao buscar funding real: {e}. Usando fallback.")
                current_funding = 0.0001

            df['funding_rate'] = current_funding

            # Aplicar indicadores
            df = FeatureEngineer.apply_indicators(df)

            # Z-score do funding
            funding_series = pd.Series([current_funding] * len(df))
            df['funding_z_score'] = (funding_series - funding_series.rolling(96).mean()) / funding_series.rolling(96).std().fillna(0)

            return df

        except Exception as e:
            print(f" Erro ao buscar dados: {e}")
            return None

    def _calculate_position_size(self, entry_price, stop_loss_price):
        """Calcula qty em BTC baseado em risco de 2% (prompt.md)"""
        if self.paper_trading:
            balance = self.paper_balance
        else:
            balance = self._get_account_balance() or 10000.0

        risk_amount = balance * self.max_risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk == 0:
            return 0.0
        qty_btc = risk_amount / price_risk  # Quantidade em BTC
        return qty_btc

    def _check_daily_drawdown(self):
        """Verifica limite de -5% no dia (prompt.md)"""
        now = datetime.now(timezone.utc)

        # Reset diario do start_balance s 00:00 UTC
        if not hasattr(self, 'last_drawdown_reset') or self.last_drawdown_reset.date() != now.date():
            if self.paper_trading:
                self.paper_start_balance = self.paper_balance
            else:
                self.daily_start_balance = self._get_account_balance()
            self.last_drawdown_reset = now
            start = self.paper_start_balance if self.paper_trading else self.daily_start_balance
            print(f" Reset diario do drawdown: ${start:.2f}" if start else " Reset diario do drawdown")

        # Obtem balances atuais
        if self.paper_trading:
            current_balance = self.paper_balance
            start_balance = self.paper_start_balance
        else:
            current_balance = self._get_account_balance()
            start_balance = self.daily_start_balance

        if current_balance is None or start_balance is None:
            return False

        drawdown = (current_balance - start_balance) / start_balance
        if drawdown <= -self.max_daily_drawdown:
            print(f" DRAWDOWN DIARIO ATINGIDO: {drawdown*100:.2f}%. SUSPENDENDO.")
            return True
        return False

    def _get_account_balance(self):
        """Busca saldo real da Binance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except Exception as e:
            print(f" Erro ao buscar saldo: {e}")
            return None
    def _check_macro_trend(self):
        """
        Busca o gráfico de 4H para definir a tendência maior.
        Retorna: 1 (Alta), -1 (Baixa), 0 (Neutro/Erro)
        """
        try:
            # Busca as últimas 50 velas de 4 Horas da Binance
            ohlcv = self.exchange_futures.fetch_ohlcv(self.symbol, timeframe='4h', limit=50)
            
            # Usando importação local do pandas para evitar erros de escopo
            import pandas as pd
            df_4h = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calcula a Média Móvel Exponencial (EMA) de 50 períodos (Tendência Institucional)
            from ta.trend import ema_indicator
            df_4h['ema_50'] = ema_indicator(df_4h['close'], window=50)
            
            last_close = df_4h['close'].iloc[-1]
            last_ema = df_4h['ema_50'].iloc[-1]
            
            # Se não houver dados suficientes para a EMA, fica neutro
            if pd.isna(last_ema):
                return 0 
                
            # Se o preço atual está acima da EMA de 4H, a maré é de ALTA. Se não, BAIXA.
            return 1 if last_close > last_ema else -1
            
        except Exception as e:
            print(f" ⚠️ Erro ao buscar tendência 4H: {e}")
            return 0 # Em caso de erro da exchange, libera o trade (Neutro)
        
    def run(self):
        mode_str = "PAPER TRADING" if self.paper_trading else "LIVE TRADING"
        print(f" Bot de Scalping BTC/USDT 15m INICIADO ({mode_str} - Binance)")
        print(f" Threshold: {self.threshold*100:.0f}%")
        print(f" Risco maximo: {self.max_risk_per_trade*100:.0f}% por operacao | Drawdown: -{self.max_daily_drawdown*100:.0f}%")

        if self.paper_trading:
            print(f" Saldo simulado: ${self.paper_balance:.2f}")
        else:
            real_balance = self._get_account_balance()
            self.daily_start_balance = real_balance
            print(f" Saldo real: ${real_balance:.2f}" if real_balance else " Saldo nao disponivel")

        while True:
            try:
                if self._check_daily_drawdown():
                    time.sleep(3600)
                    continue

                df = self._get_realtime_data()
                if df is None or df.empty:
                    print(" Sem dados, aguardando...")
                    time.sleep(900)
                    continue

                closed_candle = df.iloc[-2]
                features = closed_candle[self.features_list].values.reshape(1, -1)
                prob = self.model.predict_proba(features)[0][1]

                current_price = df.iloc[-1]['close']

                if self.finbert_available:
                    try:
                        score_sent = self._get_sentiment()
                        print(f" Sentimento FinBERT: {score_sent:.2f}")
                        prob_adj = prob * (1 + 0.15 * score_sent)
                        prob = prob_adj
                        print(f" Confianca ajustada (c/ FinBERT): {prob:.2%}")
                    except Exception as e:
                        print(f" Erro FinBERT: {e}")

                print(f" {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC | Preco: ${closed_candle['close']:.2f} | Confianca: {prob:.2%}")

                market_context = {
                    'rsi': float(closed_candle.get('RSI_norm', 0)),
                    'funding': float(closed_candle.get('funding_rate', 0)),
                    'session': 'us' if 13 <= datetime.now(timezone.utc).hour < 20 else 'other',
                    'macro_trend': float(closed_candle.get('macro_trend', 0)),
                }

                # 1. Filtro de Mercado Lateral (ADX)
                adx_value = closed_candle.get('ADX_14', 0)
                
                # 2. Filtro Multi-Timeframe (4 Horas)
                macro_trend_direction = self._check_macro_trend()

                # --- LÓGICA DE ABERTURA DE ORDEM ---
                if self.open_order is None and prob >= self.threshold and adx_value >= 20:
                    
                    if macro_trend_direction >= 0: # 1 (Alta) ou 0 (Neutro)
                        entry_price = closed_candle['close']
                        side = 'LONG'

                        # 3. Cálculo Dinâmico de SL/TP via ATR (Volatilidade)
                        current_atr = float(closed_candle.get('ATRr_14', entry_price * 0.002)) 
                        min_atr_usdt = entry_price * 0.003
                        atr_to_use = max(current_atr, min_atr_usdt)

                        atr_multiplier_sl = 1.5
                        atr_multiplier_tp = 3.0
                        
                        if side == 'LONG':
                            stop_loss = entry_price - (atr_to_use * atr_multiplier_sl)
                            take_profit = entry_price + (atr_to_use * atr_multiplier_tp)

                        qty_btc = self._calculate_position_size(entry_price, stop_loss)
                        position_size_usdt = qty_btc * entry_price 

                        limit_price = entry_price * 0.9998 
                        self.limit_order_price = limit_price
                        self.limit_order_side = side
                        self.limit_order_timestamp = datetime.now(timezone.utc)

                        self.open_order = {
                            'side': side,
                            'signal_price': entry_price, 
                            'limit_price': limit_price,
                            'take_profit': take_profit,
                            'stop_loss': stop_loss,
                            'confidence': prob,
                            'qty_btc': qty_btc,
                            'position_size_usdt': position_size_usdt,
                            'timestamp': datetime.now(timezone.utc),
                            'order_type': 'LIMIT',
                            'filled': False,
                            'actual_entry_price': None,
                            'break_even_activated': False # Estado do Break-Even
                        }

                        print(f" SINAL [{side}] | Conf: {prob:.2%} | Entrada: ${entry_price:.2f} | ADX: {adx_value:.2f}")
                        print(f" LIMIT ORDER: ${limit_price:.2f} (Maker) | TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}")
                        print(f" Posicao: ${position_size_usdt:.2f} ({qty_btc:.6f} BTC) - Risco 2%")

                    else:
                        print(f" 🛑 SINAL IGNORADO: Conflito Macro. 15m pede LONG, mas 4H esta em tendencia de BAIXA.")

                elif self.open_order is None and prob >= self.threshold and adx_value < 20:
                    print(f" ⏸️ Mercado lateral (ADX < 20) - Sinal ignorado | ADX: {adx_value:.2f} | Conf: {prob:.2%}")

                # --- LÓGICA DE GESTÃO DA ORDEM ABERTA ---
                elif self.open_order:
                    current_price = closed_candle['close']
                    order = self.open_order

                    if order.get('order_type') == 'LIMIT' and not order.get('filled', False):
                        limit_price = order['limit_price']
                        side = order['side']

                        limit_filled = False
                        if side == 'LONG' and current_price <= limit_price:
                            limit_filled = True

                        if limit_filled:
                            actual_entry_price = limit_price
                            order['filled'] = True
                            order['actual_entry_price'] = actual_entry_price
                            print(f" LIMIT ORDER PREENCHIDA em ${actual_entry_price:.2f}")
                        else:
                            time_elapsed = (datetime.now(timezone.utc) - self.limit_order_timestamp).total_seconds()
                            if time_elapsed >= 60:
                                print(f" LIMIT ORDER TIMEOUT ({time_elapsed:.0f}s) - Trocando para MARKET order")
                                market_entry_price = current_price
                                order['order_type'] = 'MARKET'
                                order['filled'] = True
                                order['actual_entry_price'] = market_entry_price
                                order['entry_price'] = market_entry_price
                                print(f" MARKET ORDER EXECUTADA em ${market_entry_price:.2f}")
                                
                                try:
                                    journal_entry = {
                                        'event': 'ORDER_FALLBACK',
                                        'timestamp': datetime.now(timezone.utc).isoformat(),
                                        'side': order['side'],
                                        'limit_price': order['limit_price'],
                                        'market_price': market_entry_price,
                                        'time_elapsed_seconds': time_elapsed,
                                        'reason': 'Timeout na ordem Limit',
                                        'paper_trading': self.paper_trading
                                    }
                                    self._log_to_journal(journal_entry)
                                except Exception as e:
                                    pass
                            else:
                                print(f"⏳ Aguardando LIMIT ordem... ({time_elapsed:.0f}s/60s) | Preco atual: ${current_price:.2f} | Limite: ${limit_price:.2f}")
                                
                                now = datetime.now(timezone.utc)
                                min_to_next_15 = 15 - (now.minute % 15)
                                next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=min_to_next_15)
                                sleep_seconds = (next_run - now).total_seconds() + 5
                                
                                import gc
                                gc.collect()
                                time.sleep(max(10, sleep_seconds)) 
                                continue

                    if order.get('filled', False) or order.get('order_type') != 'LIMIT':
                        
                        if order.get('actual_entry_price') is not None:
                            entry_price = order['actual_entry_price']
                        else:
                            entry_price = order.get('entry_price', current_price)

                        # Busca TP do dicionário da ordem (que foi calculado dinamicamente com ATR)
                        take_profit = order['take_profit']
                        
                        # 4. GATILHO DE BREAK-EVEN
                        if not order.get('break_even_activated', False):
                            if order['side'] == 'LONG':
                                distancia_tp = take_profit - entry_price
                                trigger_price = entry_price + (distancia_tp / 2.0)
                                
                                if current_price >= trigger_price:
                                    # Move para a entrada + 0.2% para cobrir taxas
                                    new_stop_loss = entry_price * (1 + self.break_even_target_pct)
                                    order['stop_loss'] = new_stop_loss
                                    order['break_even_activated'] = True
                                    print(f" 🛡️ BREAK-EVEN ATIVADO! Stop Loss movido para: ${new_stop_loss:.2f}")
                        
                        # Busca SL atualizado
                        stop_loss = order['stop_loss']

                        hit_tp = current_price >= take_profit
                        hit_sl = current_price <= stop_loss

                        if hit_tp or hit_sl:
                            result = "TP" if hit_tp else "SL"
                            price_diff = current_price - entry_price
                            gross_profit_usdt = order['qty_btc'] * price_diff

                            if order.get('order_type') == 'LIMIT' and order.get('filled', False) and not (time_elapsed >= 60 if 'time_elapsed' in locals() else False):
                                effective_fee_rate = self.fee_rate * 0.75 
                            else:
                                effective_fee_rate = self.fee_rate

                            fee_usdt = (order['qty_btc'] * entry_price * effective_fee_rate) + \
                                       (order['qty_btc'] * current_price * effective_fee_rate)

                            profit_usdt = gross_profit_usdt - fee_usdt
                            profit_pct = (profit_usdt / order['position_size_usdt']) * 100

                            if self.paper_trading:
                                self.paper_balance += profit_usdt
                                print(f" ORDEM FECHADA [{result}] | Preco: ${current_price:.2f} | PnL: {profit_pct:.2f}% (${profit_usdt:.2f})")
                                print(f" Saldo simulado: ${self.paper_balance:.2f}")
                            else:
                                print(f" ORDEM FECHADA [{result}] | Preco: ${current_price:.2f} | PnL: {profit_pct:.2f}%")

                            self.db_manager.execute_query(
                                "UPDATE trade_history SET status = ? WHERE id = (SELECT MAX(id) FROM trade_history)",
                                (result,)
                            )

                            close_journal = {
                                'event': 'CLOSE',
                                'timestamp': datetime.now(timezone.utc).isoformat(),
                                'result': result,
                                'exit_price': current_price,
                                'profit_pct': profit_pct,
                                'profit_usdt': profit_usdt if self.paper_trading else None,
                                'paper_balance_after': self.paper_balance if self.paper_trading else None,
                                'reason': f"Hit {result} (Tipo: {order.get('order_type', 'UNKNOWN')} | ATR adaptativo)",
                                'paper_trading': self.paper_trading
                            }
                            self._log_to_journal(close_journal)

                            self.open_order = None
                            self.trade_count += 1

                # --- SINCRONIZAÇÃO DE TEMPO ABSOLUTA ---
                now = datetime.now(timezone.utc)
                min_to_next_15 = 15 - (now.minute % 15)
                next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=min_to_next_15)
                sleep_seconds = (next_run - now).total_seconds() + 5 
                
                print(f"⏳ Dormindo por {sleep_seconds/60:.2f} minutos. Proxima leitura as {next_run.strftime('%H:%M:%S')} UTC...")
                
                import gc
                gc.collect()
                time.sleep(max(10, sleep_seconds))

            except KeyboardInterrupt:
                print("\n Bot encerrado pelo usuario.")
                break
            except Exception as e:
                print(f" Erro no loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='BTC Scalping Bot 15m')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--paper', action='store_true', help='Paper Trading (simulacao, default)')
    group.add_argument('--live', action='store_true', help='Executar com capital real')
    parser.add_argument('--threshold', type=float, default=0.60, help='Confianca minima (default: 0.60)')
    args = parser.parse_args()

    # Default: Paper Trading (se nenhum modo especificado ou se --paper)
    paper_trading = not args.live
    bot = TradingBot(threshold=args.threshold, paper_trading=paper_trading)
    bot.run()
