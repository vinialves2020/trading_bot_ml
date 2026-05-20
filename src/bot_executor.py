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
                'defaultType': 'future',
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

        # Carregar Modelos
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(base_path, "data", "models_weights")

        model_path = os.path.join(model_dir, "xgb_oraculo_btc.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f" Modelo nao encontrado em: {model_path}")
        self.model = XGBClassifier()
        self.model.load_model(model_path)

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

        self.kelly_fraction = 0.02
        metrics_path = os.path.join(base_path, "data", "training_metrics.json")
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)
                self.kelly_fraction = metrics.get('kelly_fraction') or 0.02
                print(f" Kelly Criterion: {self.kelly_fraction*100:.2f}% (do treino)")

        self.tp_pct = 0.006   # 0.6%
        self.sl_pct = 0.003   # 0.3%
        self.break_even_trigger_pct = 0.0045 # 0.45% de lucro ativa a proteção
        self.break_even_target_pct = 0.0010 # Move o stop para a entrada + 0.1%
        self.fee_rate = 0.001 # 0.1% Taxa
        self.max_risk_per_trade = self.kelly_fraction
        self.max_daily_drawdown = 0.05

        self.daily_start_balance = None
        self.open_order = None
        self.paper_balance = 100.0
        self.paper_start_balance = 100.0
        self.trade_count = 0
        
        self.order_timestamp = None
        self.order_id = None
        
        self.limit_order_price = None
        self.limit_order_side = None
        self.limit_order_timestamp = None

        self.journal_file = os.path.join(base_path, "data", "trade_journal.jsonl")

        self._init_log_table()

    def _init_log_table(self):
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
        with open(self.journal_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trade_data, default=str) + '\n')

    def _get_realtime_data(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='15m', limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            try:
                funding = self.exchange_futures.fetch_funding_rate_history(
                    symbol='BTC/USDT:USDT', limit=1
                )
                current_funding = funding[0]['fundingRate'] if funding else 0.0001
            except Exception as e:
                print(f" Erro ao buscar funding real: {e}. Usando fallback.")
                current_funding = 0.0001

            df['funding_rate'] = current_funding
            df = FeatureEngineer.apply_indicators(df)
            
            funding_series = pd.Series([current_funding] * len(df))
            df['funding_z_score'] = (funding_series - funding_series.rolling(96).mean()) / funding_series.rolling(96).std().fillna(0)

            return df
        except Exception as e:
            print(f" Erro ao buscar dados: {e}")
            return None

    def _calculate_position_size(self, entry_price, stop_loss_price):
        if self.paper_trading:
            balance = self.paper_balance
        else:
            balance = self._get_account_balance() or 10000.0

        risk_amount = balance * self.max_risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk == 0:
            return 0.0
        qty_btc = risk_amount / price_risk
        return qty_btc

    def _check_daily_drawdown(self):
        now = datetime.now(timezone.utc)
        if not hasattr(self, 'last_drawdown_reset') or self.last_drawdown_reset.date() != now.date():
            if self.paper_trading:
                self.paper_start_balance = self.paper_balance
            else:
                self.daily_start_balance = self._get_account_balance()
            self.last_drawdown_reset = now
            start = self.paper_start_balance if self.paper_trading else self.daily_start_balance
            print(f" Reset diario do drawdown: ${start:.2f}" if start else " Reset diario do drawdown")

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
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except Exception as e:
            print(f" Erro ao buscar saldo: {e}")
            return None

    def _check_macro_trend(self):
        try:
            ohlcv = self.exchange_futures.fetch_ohlcv(self.symbol, timeframe='4h', limit=50)
            import pandas as pd
            df_4h = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            from ta.trend import ema_indicator
            df_4h['ema_50'] = ema_indicator(df_4h['close'], window=50)
            last_close = df_4h['close'].iloc[-1]
            last_ema = df_4h['ema_50'].iloc[-1]
            if pd.isna(last_ema):
                return 0 
            return 1 if last_close > last_ema else -1
        except Exception as e:
            print(f" Aviso ao buscar tendencia 4H: {e}")
            return 0
        
    def run(self):
        mode_str = "PAPER TRADING" if self.paper_trading else "LIVE TRADING (SIMULATION)"
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

                if self.open_order is None:
                    # ESTADO A: Buscando Sinal (Apenas no fechamento de cada 15m)
                    now = datetime.now(timezone.utc)
                    min_to_next_15 = 15 - (now.minute % 15)
                    next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=min_to_next_15)
                    sleep_seconds = (next_run - now).total_seconds() + 5 
                    
                    if sleep_seconds > 10:
                        print(f"⏳ Aguardando proximo candle de 15m... Dormindo {sleep_seconds/60:.2f} min.")
                        import gc
                        gc.collect()
                        time.sleep(sleep_seconds)
                        continue # Recomeca o loop pra checar drawdown

                    # Acordou, processa as features
                    df = self._get_realtime_data()
                    if df is None or df.empty:
                        print(" Sem dados, aguardando...")
                        time.sleep(60)
                        continue

                    closed_candle = df.iloc[-2]
                    features = closed_candle[self.features_list].values.reshape(1, -1)
                    prob = self.model.predict_proba(features)[0][1]

                    adx_value = closed_candle.get('ADX_14', 0)
                    macro_trend_direction = self._check_macro_trend()

                    print(f" {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC | Preco Fechamento: ${closed_candle['close']:.2f} | Confianca Base: {prob:.2%}")

                    # FinBERT so e acionado se os filtros basicos passarem e a prob base estiver perto do gatilho 
                    min_prob_needed = self.threshold / 1.15
                    if prob >= min_prob_needed and adx_value >= 20 and macro_trend_direction >= 0:
                        if self.finbert_available:
                            try:
                                print(f" 🤖 Confianca na zona de gatilho ({prob:.2%}). Invocando FinBERT...")
                                score_sent = self._get_sentiment()
                                prob_adj = prob * (1 + 0.15 * score_sent)
                                prob = prob_adj
                                print(f" Confianca ajustada (c/ FinBERT): {prob:.2%}")
                            except Exception as e:
                                pass

                    if prob >= self.threshold and adx_value >= 20:
                        if macro_trend_direction >= 0:
                            entry_price = closed_candle['close']
                            side = 'LONG'

                            current_atr = float(closed_candle.get('ATRr_14', entry_price * 0.002)) 
                            min_atr_usdt = entry_price * 0.003
                            atr_to_use = max(current_atr, min_atr_usdt)

                            stop_loss = entry_price - (atr_to_use * 1.5)
                            take_profit = entry_price + (atr_to_use * 3.0)

                            qty_btc = self._calculate_position_size(entry_price, stop_loss)
                            position_size_usdt = qty_btc * entry_price 

                            limit_price = entry_price * 0.9998 
                            
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
                                'limit_order_timestamp': datetime.now(timezone.utc),
                                'order_type': 'LIMIT',
                                'filled': False,
                                'actual_entry_price': None,
                                'break_even_activated': False
                            }

                            print(f" SINAL [{side}] | Conf: {prob:.2%} | Limit Maker: ${limit_price:.2f} | TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}")
                            
                            # Pequeno delay antes de transicionar pro Estado B
                            time.sleep(5)
                        else:
                            print(f" 🛑 SINAL IGNORADO: Conflito Macro (Baixa).")
                    elif prob >= self.threshold and adx_value < 20:
                        print(f" ⏸️ Mercado lateral (ADX < 20) - Sinal ignorado | Conf: {prob:.2%}")
                    else:
                        # Nenhum sinal, apenas espera o próximo candle no próximo loop
                        pass

                else:
                    # ESTADO B: Monitoramento Ativo de Ordem (Tick a Tick a cada 5s)
                    try:
                        ticker = self.exchange.fetch_ticker(self.symbol)
                        current_price = ticker['last']
                    except Exception as e:
                        print(f"Erro ao buscar ticker: {e}")
                        time.sleep(5)
                        continue

                    order = self.open_order
                    
                    # 1. Gerencia Preenchimento da Ordem Limit
                    if order['order_type'] == 'LIMIT' and not order['filled']:
                        limit_price = order['limit_price']
                        
                        if order['side'] == 'LONG' and current_price <= limit_price:
                            order['filled'] = True
                            order['actual_entry_price'] = limit_price
                            print(f"✅ LIMIT ORDER PREENCHIDA em ${limit_price:.2f}!")
                        else:
                            time_elapsed = (datetime.now(timezone.utc) - order['limit_order_timestamp']).total_seconds()
                            if time_elapsed >= 60:
                                print(f"⏳ LIMIT ORDER TIMEOUT (60s) - Trocando para MARKET order em ${current_price:.2f}")
                                order['order_type'] = 'MARKET'
                                order['filled'] = True
                                order['actual_entry_price'] = current_price
                                
                                try:
                                    self._log_to_journal({
                                        'event': 'ORDER_FALLBACK',
                                        'timestamp': datetime.now(timezone.utc).isoformat(),
                                        'market_price': current_price,
                                        'reason': 'Timeout Limit',
                                    })
                                except: pass
                            else:
                                # Ainda esperando preencher
                                time.sleep(5)
                                continue

                    # 2. Gerencia Alvos e Break-Even (Apenas se a ordem está preenchida)
                    if order['filled']:
                        entry_price = order['actual_entry_price']
                        take_profit = order['take_profit']
                        stop_loss = order['stop_loss']
                        
                        # Monitoramento Break-Even
                        if not order['break_even_activated'] and order['side'] == 'LONG':
                            trigger_price = entry_price * (1 + self.break_even_trigger_pct)
                            if current_price >= trigger_price:
                                new_stop_loss = entry_price * (1 + self.break_even_target_pct)
                                order['stop_loss'] = new_stop_loss
                                order['break_even_activated'] = True
                                print(f"🛡️ BREAK-EVEN ATIVADO! Stop Loss subiu para: ${new_stop_loss:.2f}")
                                
                        # Atualiza Stop_loss caso Break-Even ativado
                        stop_loss = order['stop_loss']

                        hit_tp = current_price >= take_profit
                        hit_sl = current_price <= stop_loss

                        if hit_tp or hit_sl:
                            result = "TP" if hit_tp else "SL"
                            price_diff = current_price - entry_price
                            gross_profit_usdt = order['qty_btc'] * price_diff

                            if order['order_type'] == 'LIMIT':
                                effective_fee_rate = self.fee_rate * 0.75 # Maker entry, taker exit
                            else:
                                effective_fee_rate = self.fee_rate # Taker entry, taker exit

                            fee_usdt = (order['qty_btc'] * entry_price * effective_fee_rate) + \
                                       (order['qty_btc'] * current_price * effective_fee_rate)

                            profit_usdt = gross_profit_usdt - fee_usdt
                            profit_pct = (profit_usdt / order['position_size_usdt']) * 100

                            if self.paper_trading:
                                self.paper_balance += profit_usdt
                                print(f"💰 ORDEM FECHADA [{result}] | Preco: ${current_price:.2f} | PnL Liquido: {profit_pct:.2f}% (${profit_usdt:.2f})")
                                print(f" Saldo simulado atual: ${self.paper_balance:.2f}")
                            else:
                                print(f"💰 ORDEM FECHADA [{result}] | Preco: ${current_price:.2f} | PnL Liquido: {profit_pct:.2f}%")

                            self.db_manager.execute_query(
                                "UPDATE trade_history SET status = ? WHERE id = (SELECT MAX(id) FROM trade_history)",
                                (result,)
                            )

                            try:
                                self._log_to_journal({
                                    'event': 'CLOSE',
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
                                    'result': result,
                                    'exit_price': current_price,
                                    'profit_pct': profit_pct,
                                    'profit_usdt': profit_usdt if self.paper_trading else None,
                                    'paper_balance': self.paper_balance if self.paper_trading else None
                                })
                            except: pass

                            # Limpa a ordem para voltar pro Estado A
                            self.open_order = None
                            self.trade_count += 1
                        
                        else:
                            # Ordem segue aberta. Dorme curto para monitoramento tick-a-tick.
                            time.sleep(5)

            except KeyboardInterrupt:
                print("\n Bot encerrado pelo usuario.")
                break
            except Exception as e:
                print(f" Erro no loop principal: {e}")
                time.sleep(15)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='BTC Scalping Bot 15m')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--paper', action='store_true', help='Paper Trading (simulacao, default)')
    group.add_argument('--live', action='store_true', help='Executar com capital real (Simulado sem envio da API)')
    parser.add_argument('--threshold', type=float, default=0.60, help='Confianca minima (default: 0.60)')
    args = parser.parse_args()

    paper_trading = not args.live
    bot = TradingBot(threshold=args.threshold, paper_trading=paper_trading)
    bot.run()
