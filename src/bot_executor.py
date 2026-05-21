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

# ── Duplica stdout para arquivo de log legível pelo Dashboard ────────────────
class _Tee:
    """Escreve no stdout E num arquivo de log ao mesmo tempo, com rotação simples."""
    MAX_BYTES = 512 * 1024  # 512 KB por log

    def __init__(self, stream, filepath):
        self._stream   = stream
        self._filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._f = open(filepath, 'a', encoding='utf-8')

    def write(self, data):
        try: self._stream.write(data)
        except Exception: pass
        try:
            # Rotação: se o arquivo cresceu demais, mantém apenas a metade mais recente
            if os.path.getsize(self._filepath) > self.MAX_BYTES:
                self._f.close()
                with open(self._filepath, 'r', encoding='utf-8', errors='replace') as rf:
                    lines = rf.readlines()
                with open(self._filepath, 'w', encoding='utf-8') as wf:
                    wf.writelines(lines[len(lines) // 2:])
                self._f = open(self._filepath, 'a', encoding='utf-8')
            self._f.write(data)
            self._f.flush()
        except Exception: pass

    def flush(self):
        try: self._stream.flush()
        except Exception: pass
        try: self._f.flush()
        except Exception: pass


class TradingBot:
    def __init__(self, symbol='BTC/USDT', timeframe='15m', threshold=0.60, paper_trading=True):
        self.symbol = symbol
        self.timeframe = timeframe
        self.prefix = f"{symbol.split('/')[0].lower()}_{timeframe}"
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
            from src.models.finbert_sentiment import FinBERTSentiment
            sentiment_client = FinBERTSentiment()
            self._get_sentiment = lambda: sentiment_client.analisar_sentimento(self.symbol)
            self.finbert_available = True
            print(" FinBERT disponvel para anlise de sentimento (via API)")
        except ImportError:
            print(" Aviso: FinBERT no disponvel. Instale: pip install transformers torch")

        # Carregar Modelos
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(base_path, "data", "models_weights")

        model_path = os.path.join(model_dir, f"xgb_oraculo_{self.prefix}.json")
        if not os.path.exists(model_path):
            fallback_path = os.path.join(model_dir, f"xgb_oraculo_{symbol.split('/')[0].lower()}.json")
            if os.path.exists(fallback_path):
                model_path = fallback_path
            else:
                raise FileNotFoundError(f" Modelo nao encontrado em: {model_path}")
        self.model = XGBClassifier(n_jobs=1)  # Limita threads para economizar RAM
        self.model.load_model(model_path)

        magnitude_path = os.path.join(model_dir, f"lgbm_magnitude_{self.prefix}.txt")
        if not os.path.exists(magnitude_path):
            fallback_mag_path = os.path.join(model_dir, f"lgbm_magnitude_{symbol.split('/')[0].lower()}.txt")
            if os.path.exists(fallback_mag_path):
                magnitude_path = fallback_mag_path
                
        self.magnitude_available = False
        if os.path.exists(magnitude_path):
            try:
                self.model_magnitude = lgb.Booster(model_file=magnitude_path)
                self.magnitude_available = True
                print(" Modelo de Magnitude (LightGBM) carregado")
            except Exception as e:
                print(f" Aviso: Magnitude nao carregada: {e}")

        self.features_list = FeatureEngineer.get_feature_list()

        # ── WARM-UP (Aquecimento da IA) ──
        # Força o XGBoost a alocar seus buffers de memória agora, para não crashar após o primeiro sleep
        try:
            print(" 🧠 Realizando aquecimento do modelo (Dummy Inference)...")
            dummy_features = np.zeros((1, len(self.features_list)))
            self.model.predict_proba(dummy_features)
            print(" ✅ Aquecimento concluído. Memória alocada com sucesso.")
        except Exception as e:
            print(f" Aviso durante aquecimento: {e}")

        self.kelly_fraction = 0.02
        metrics_path = os.path.join(base_path, "data", f"training_metrics_{self.prefix}.json")
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)
                self.kelly_fraction = metrics.get('kelly_fraction') or 0.02
                print(f" Kelly Criterion: {self.kelly_fraction*100:.2f}% (do treino)")

        if self.timeframe == '1h':
            self.break_even_trigger_pct = 0.010
            self.break_even_target_pct = 0.002
        else:
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

        # ── Redireciona stdout para arquivo de log (legível pelo Dashboard) ───
        log_dir  = os.path.join(base_path, "data", "logs")
        sym_clean = self.symbol.replace("/", "_")
        log_path  = os.path.join(log_dir, f"bot_{sym_clean}.log")
        sys.stdout = _Tee(sys.__stdout__, log_path)

        self._init_log_table()

    def _init_log_table(self):
        # USA 'IF NOT EXISTS' para NAO apagar os trades dos outros bots ao reiniciar
        query = """
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timeframe TEXT,
            timestamp DATETIME,
            side TEXT,
            entry_price REAL,
            take_profit REAL,
            stop_loss REAL,
            confidence REAL,
            position_size_usdt REAL,
            result TEXT,
            profit_pct REAL,
            profit_usdt REAL,
            paper_balance_after REAL,
            event TEXT DEFAULT 'ENTRY'
        )
        """
        self.db_manager.execute_query(query)

        # ── Migração automática: adiciona colunas ausentes em bancos antigos ──
        import sqlite3 as _sq
        _db = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'trading_data.db'))
        try:
            _conn = _sq.connect(_db)
            existing = {row[1] for row in _conn.execute("PRAGMA table_info(trade_history)")}
            novas = {
                "symbol":              "TEXT",
                "timeframe":           "TEXT",
                "result":              "TEXT",
                "profit_pct":          "REAL",
                "profit_usdt":         "REAL",
                "paper_balance_after": "REAL",
                "event":               "TEXT DEFAULT 'ENTRY'",
            }
            for col, tipo in novas.items():
                if col not in existing:
                    _conn.execute(f"ALTER TABLE trade_history ADD COLUMN {col} {tipo}")
                    print(f"  ↳ DB Migração: coluna '{col}' adicionada.")
            _conn.commit()
            _conn.close()
        except Exception as _e:
            print(f"  Aviso migração DB: {_e}")

    def _log_to_journal(self, trade_data):
        with open(self.journal_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trade_data, default=str) + '\n')

    def _get_realtime_data(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=1000)
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
        print(f" Bot de Scalping {self.symbol} {self.timeframe} INICIADO ({mode_str} - Binance)")
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
                    # ESTADO A: Buscando Sinal
                    now = datetime.now(timezone.utc)
                    if self.timeframe == '1h':
                        min_to_next = 60 - now.minute
                    else:
                        min_to_next = 15 - (now.minute % 15)
                    next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=min_to_next)
                    sleep_seconds = (next_run - now).total_seconds() + 5 
                    
                    if sleep_seconds > 0:
                        print(f"⏳ Aguardando proximo candle de {self.timeframe}... Dormindo {sleep_seconds/60:.2f} min.")
                        import gc
                        gc.collect()
                        time.sleep(sleep_seconds)

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

                            if self.timeframe == '1h':
                                stop_loss = entry_price * (1 - 0.0075)
                                take_profit = entry_price * (1 + 0.015)
                            else:
                                stop_loss = entry_price * (1 - 0.0045)
                                take_profit = entry_price * (1 + 0.009)

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

                            # Registra o sinal de entrada no banco com o simbolo correto
                            self.db_manager.execute_query(
                                """INSERT INTO trade_history
                                   (symbol, timeframe, timestamp, side, entry_price, take_profit, stop_loss, confidence, position_size_usdt, event)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ENTRY')""",
                                (self.symbol, self.timeframe, datetime.now(timezone.utc).isoformat(),
                                 side, entry_price, take_profit, stop_loss, prob, position_size_usdt)
                            )

                            try:
                                self._log_to_journal({
                                    'event': 'ENTRY',
                                    'symbol': self.symbol,
                                    'timeframe': self.timeframe,
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
                                    'side': side,
                                    'entry_price': entry_price,
                                    'limit_price': limit_price,
                                    'take_profit': take_profit,
                                    'stop_loss': stop_loss,
                                    'confidence': prob,
                                    'position_size_usdt': position_size_usdt,
                                })
                            except: pass

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
                                        'symbol': self.symbol,
                                        'timeframe': self.timeframe,
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

                            # Atualiza o registro de ENTRY com os dados do fechamento
                            self.db_manager.execute_query(
                                """UPDATE trade_history SET result = ?, profit_pct = ?, profit_usdt = ?,
                                   paper_balance_after = ?, event = 'CLOSE'
                                   WHERE id = (SELECT MAX(id) FROM trade_history WHERE symbol = ?)""",
                                (result, profit_pct,
                                 profit_usdt if self.paper_trading else None,
                                 self.paper_balance if self.paper_trading else None,
                                 self.symbol)
                            )

                            try:
                                self._log_to_journal({
                                    'event': 'CLOSE',
                                    'symbol': self.symbol,
                                    'timeframe': self.timeframe,
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
                                    'result': result,
                                    'exit_price': current_price,
                                    'profit_pct': profit_pct,
                                    'profit_usdt': profit_usdt if self.paper_trading else None,
                                    'paper_balance_after': self.paper_balance if self.paper_trading else None
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
    parser = argparse.ArgumentParser(description='Multi-Coin Trading Bot')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Simbolo para operar (ex: BTC/USDT, ETH/USDT)')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe (ex: 15m, 1h)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--paper', action='store_true', help='Paper Trading (simulacao, default)')
    group.add_argument('--live', action='store_true', help='Executar com capital real (Simulado sem envio da API)')
    parser.add_argument('--threshold', type=float, default=0.53, help='Confianca minima (default: 0.53)')
    args = parser.parse_args()

    paper_trading = not args.live
    bot = TradingBot(symbol=args.symbol, timeframe=args.timeframe, threshold=args.threshold, paper_trading=paper_trading)
    bot.run()
