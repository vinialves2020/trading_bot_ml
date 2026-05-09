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
    def __init__(self, symbol='BTC/USDT', threshold=0.50, paper_trading=True):
        self.symbol = symbol
        self.threshold = threshold
        self.paper_trading = paper_trading
        self.db_manager = DatabaseManager('data/trading_data.db')

       # Conexões Bybit (Substituto para burlar o bloqueio de IP da AWS)
        self.exchange = ccxt.bybit({'enableRateLimit': True})
        
        # O equivalente ao binanceusdm (Futuros) na Bybit é o tipo 'linear'
        self.exchange_futures = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'} 
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
        self.tp_pct = 0.004   # 0.4%
        self.sl_pct = 0.002   # 0.2%
        self.fee_rate = 0.001 # 0.1% Taxa da Binance (Maker/Taker)
        self.max_risk_per_trade = self.kelly_fraction  # Usa Kelly dinamico
        self.max_daily_drawdown = 0.05   # -5% suspende

        # Estado do bot
        self.daily_start_balance = None
        self.open_order = None
        self.paper_balance = 10000.0  # Saldo simulado R$ 10.000
        self.paper_start_balance = 10000.0
        self.trade_count = 0

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

    def run(self):
        mode_str = "PAPER TRADING" if self.paper_trading else "LIVE TRADING"
        print(f" Bot de Scalping BTC/USDT 15m INICIADO ({mode_str} - Binance)")
        print(f" Threshold: {self.threshold*100:.0f}% | TP: {self.tp_pct*100:.1f}% | SL: {self.sl_pct*100:.1f}%")
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

                # 1. ORÁCULO: Olha para a última vela FECHADA (índice -2) para garantir a integridade matemática
                closed_candle = df.iloc[-2]
                features = closed_candle[self.features_list].values.reshape(1, -1)
                prob = self.model.predict_proba(features)[0][1]

                # 2. MERCADO: O preço que compramos agora é o da vela ATUAL (índice -1)
                current_price = df.iloc[-1]['close']

                # FinBERT Sentimento (opcional, enriquece contexto)
                if self.finbert_available:
                    try:
                        score_sent = self._get_sentiment()
                        debug_msg = f" Sentimento FinBERT: {score_sent:.2f}"
                        print(debug_msg)
                        # Ajustar confiana baseado em sentimento (finbert_training_prompt.md)
                        prob_adj = prob * (1 + 0.15 * score_sent)
                        prob = prob_adj  # Aplica o ajuste na deciso
                        print(f" Confiana ajustada (c/ FinBERT): {prob:.2%}")
                    except Exception as e:
                        print(f" Erro FinBERT: {e}")

                # Debug: mostra a confiana a cada ciclo
                print(f" {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC | Preo: ${closed_candle['close']:.2f} | Confiana: {prob:.2%}")

                # Contexto de mercado para o Journal
                market_context = {
                    'rsi': float(closed_candle.get('RSI_norm', 0)),
                    'funding': float(closed_candle.get('funding_rate', 0)),
                    'session': 'us' if 13 <= datetime.now(timezone.utc).hour < 20 else 'other',
                    'macro_trend': float(closed_candle.get('macro_trend', 0)),
                }

                if self.open_order is None and prob >= self.threshold:
                    entry_price = closed_candle['close']
                    side = 'LONG'

                    take_profit = entry_price * (1 + self.tp_pct)
                    stop_loss = entry_price * (1 - self.sl_pct)

                    # CALCULO DE TAMANHO DA POSICAO (2% risco)
                    # Retorna qty em BTC
                    qty_btc = self._calculate_position_size(entry_price, stop_loss)
                    position_size_usdt = qty_btc * entry_price  # Valor em USDT

                    self.open_order = {
                        'side': side,
                        'entry_price': entry_price,
                        'take_profit': take_profit,
                        'stop_loss': stop_loss,
                        'confidence': prob,
                        'qty_btc': qty_btc,
                        'position_size_usdt': position_size_usdt,
                        'timestamp': datetime.now(timezone.utc)
                    }

                    print(f" SINAL [{side}] | Conf: {prob:.2%} | Entrada: ${entry_price:.2f}")
                    print(f" TP: ${take_profit:.2f} | SL: ${stop_loss:.2f}")
                    print(f" Posicao: ${position_size_usdt:.2f} ({qty_btc:.6f} BTC) - Risco 2%")

                    # SQLite Log
                    try:
                        self.db_manager.execute_query(
                            "INSERT INTO trade_history (timestamp, side, entry_price, take_profit, stop_loss, confidence, position_size_usdt) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (datetime.now(timezone.utc), side, entry_price, take_profit, stop_loss, prob, position_size_usdt)
                        )
                        print(" SQLite: Trade salvo")
                    except Exception as e:
                        print(f" SQLite ERRO: {e}")

                    # JSONL Trade Journal (prompt.md)
                    try:
                        journal_entry = {
                            'event': 'ENTRY',
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'side': side,
                            'entry_price': entry_price,
                            'take_profit': take_profit,
                            'stop_loss': stop_loss,
                            'confidence': prob,
                            'position_size_usdt': position_size_usdt,
                            'position_qty_btc': qty_btc,
                            'market_context': market_context,
                            'reason': f"XGBoost prediction {prob:.2%} >= threshold {self.threshold:.2%}",
                            'paper_trading': self.paper_trading
                        }
                        self._log_to_journal(journal_entry)
                        print(" JSONL: Trade salvo")
                    except Exception as e:
                        print(f" JSONL ERRO: {e}")

                elif self.open_order:
                    current_price = closed_candle['close']
                    order = self.open_order

                    hit_tp = current_price >= order['take_profit']
                    hit_sl = current_price <= order['stop_loss']

                    if hit_tp or hit_sl:
                        result = "TP" if hit_tp else "SL"
                        # PnL bruto
                        price_diff = current_price - order['entry_price']
                        gross_profit_usdt = order['qty_btc'] * price_diff
                        
                        # Calculo EXATO das taxas (Compra + Venda)
                        fee_usdt = (order['qty_btc'] * order['entry_price'] * self.fee_rate) + \
                                   (order['qty_btc'] * current_price * self.fee_rate)
                        
                        # PnL Liquido (O dinheiro que realmente vai pro seu bolso)
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

                        # JSONL Close Journal
                        close_journal = {
                            'event': 'CLOSE',
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'result': result,
                            'exit_price': current_price,
                            'profit_pct': profit_pct,
                            'profit_usdt': profit_usdt if self.paper_trading else None,
                            'paper_balance_after': self.paper_balance if self.paper_trading else None,
                            'reason': f"Price hit {result}",
                            'paper_trading': self.paper_trading
                        }
                        self._log_to_journal(close_journal)

                        self.open_order = None
                        self.trade_count += 1

                # --- SINCRONIZAÇÃO DE TEMPO ABSOLUTA ---
                now = datetime.now(timezone.utc)
                # Quantos minutos faltam para a próxima janela de 15m (ex: 15:00, 15:15, 15:30)
                min_to_next_15 = 15 - (now.minute % 15)
                
                # Calcula a hora exata da próxima execução e zera os segundos
                next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=min_to_next_15)
                
                # Adiciona 5 segundos de "Gordura" para garantir que a Binance já fechou a vela no servidor deles
                sleep_seconds = (next_run - now).total_seconds() + 5 
                
                print(f"⏳ Dormindo por {sleep_seconds/60:.2f} minutos. Próxima leitura às {next_run.strftime('%H:%M:%S')} UTC...")
                
                # Coleta de lixo forçada para limpar a RAM antes de dormir
                import gc
                gc.collect()
                
                time.sleep(max(10, sleep_seconds)) # Garante que nunca durma menos de 10s em caso de lag

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
    parser.add_argument('--threshold', type=float, default=0.50, help='Confianca minima (default: 0.50)')
    args = parser.parse_args()

    # Default: Paper Trading (se nenhum modo especificado ou se --paper)
    paper_trading = not args.live
    bot = TradingBot(threshold=args.threshold, paper_trading=paper_trading)
    bot.run()
