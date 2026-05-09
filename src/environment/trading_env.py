import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BitcoinTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000.0, commission=0.001, mode='aggressive'):
        super(BitcoinTradingEnv, self).__init__()
        self.df = df
        self.mode = mode
        self.initial_balance = initial_balance
        self.commission = commission

        self.position_size = 0.20
        self.stop_loss_pct  = 0.02   # ✅ CORRIGIDO: arriscamos 2%
        self.take_profit_pct = 0.04  # ✅ CORRIGIDO: para ganhar 4% (ratio 1:2)

        self.action_space = spaces.Discrete(3)  # 0: Neutro, 1: Long, 2: Short

        # ── Features de mercado (sincronizadas com FeatureEngineer.get_feature_list())
        self.features_seguras = [
            'ret_1h', 'ret_4h', 'ret_12h', 'ret_24h', 'momentum_accel',
            'macro_trend', 'dist_ema9', 'dist_ema21',
            'price_position_24h', 'price_position_4h',
            'volume_ratio_24h', 'volume_ratio_4h',
            'RSI_norm',
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'atr_pct', 'BBP_20_2.0_2.0', 'BBB_20_2.0_2.0',
        ]
        N_MARKET_FEATURES   = len(self.features_seguras)  # 19
        N_POSITION_FEATURES = 3  # [side_norm, pnl_pct, steps_in_trade_norm]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(N_MARKET_FEATURES + N_POSITION_FEATURES,),
            dtype=np.float32
        )

    # ────────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Curriculum learning: ponto de início aleatório a cada episódio
        # Garante pelo menos 500 candles de episódio
        max_start = max(0, len(self.df) - 500)
        self.current_step = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0

        self.balance       = self.initial_balance
        self.net_worth     = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance

        # Estado da posição
        self.position_side    = 0    # 0: Neutro, 1: Long, 2: Short
        self.entry_price      = 0.0
        self.qty              = 0.0
        self.net_worth_at_entry = self.initial_balance
        self.steps_in_trade   = 0
        self.steps_neutral    = 0
        self.prev_action      = 0
        self.prev_position_side = 0

        return self._get_observation(), {}

    # ────────────────────────────────────────────────────────────────────────────
    def _get_observation(self):
        obs_market = self.df.iloc[self.current_step][self.features_seguras].values.astype(np.float32)

        # Features de estado da posição (o agente sabe o que está fazendo)
        current_price = self.df.iloc[self.current_step]['close']
        pnl_pct = 0.0
        if self.position_side == 1 and self.entry_price > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        elif self.position_side == 2 and self.entry_price > 0:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        max_steps = len(self.df)
        position_state = np.array([
            self.position_side / 2.0,                        # [0, 0.5, 1.0]
            np.clip(pnl_pct, -0.1, 0.1),                    # P&L não realizado, clipado
            min(self.steps_in_trade / 100.0, 1.0),          # tempo na posição
        ], dtype=np.float32)

        return np.concatenate([obs_market, position_state])

    # ────────────────────────────────────────────────────────────────────────────
    def step(self, action):
        self.prev_net_worth     = self.net_worth
        self.prev_position_side = self.position_side

        current_price = self.df.iloc[self.current_step]['close']
        current_atr   = self.df.iloc[self.current_step]['ATRr_14']
        atr_pct       = current_atr / current_price

        # Stop loss e take profit dinâmicos baseados em ATR
        dynamic_sl = max(atr_pct * 1.5, self.stop_loss_pct)   # mínimo 2%
        dynamic_tp = max(atr_pct * 3.0, self.take_profit_pct) # mínimo 4%, ratio ~1:2

        # ── Verificar SL/TP ─────────────────────────────────────────────────────
        if self.position_side == 1:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            if profit_pct <= -dynamic_sl or profit_pct >= dynamic_tp:
                action = 0

        elif self.position_side == 2:
            profit_pct = (self.entry_price - current_price) / self.entry_price
            if profit_pct <= -dynamic_sl or profit_pct >= dynamic_tp:
                action = 0

        # ── Execução ────────────────────────────────────────────────────────────
        if action != self.position_side or action == 0:
            self._close_position(current_price)

        if self.mode == 'conservative' and action == 2:
            action = 0

        if action != 0 and self.position_side == 0:
            self._open_position(action, current_price)

        # ── Atualizar contador de tempo na posição ───────────────────────────────
        if self.position_side != 0:
            self.steps_in_trade += 1
            self.steps_neutral   = 0
        else:
            self.steps_in_trade  = 0
            self.steps_neutral  += 1

        self.current_step += 1
        self._update_net_worth(current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # ── REWARD ESPARSO: só no fechamento de posição ─────────────────────────
        reward = 0.0

        position_just_closed = (self.prev_position_side != 0 and self.position_side == 0)
        if position_just_closed:
            trade_return = (self.net_worth - self.net_worth_at_entry) / self.net_worth_at_entry
            reward = trade_return * 100.0  # escala: ±4% vira ±4.0 de reward

        # Penalidade leve por inação prolongada (evita política 100% neutro)
        if self.position_side == 0 and self.steps_neutral > 200:
            reward -= 0.0005

        self.prev_action = action

        info = {
            'step': self.current_step,
            'net_worth': self.net_worth,
            'position_side': self.position_side,
            'action': int(action),
        }

        terminated = self.current_step >= len(self.df) - 1
        truncated  = self.net_worth <= self.initial_balance * 0.5
        return self._get_observation(), reward, terminated, truncated, info

    # ────────────────────────────────────────────────────────────────────────────
    def _open_position(self, side, price):
        amount = self.net_worth * self.position_size
        self.qty = (amount * (1 - self.commission)) / price
        self.balance -= amount
        self.entry_price = price
        self.position_side = side
        self.net_worth_at_entry = self.net_worth  # snapshot para reward esparso
        self.steps_in_trade = 0

    def _close_position(self, price):
        if self.position_side == 1:
            revenue = self.qty * price
            self.balance += revenue * (1 - self.commission)
        elif self.position_side == 2:
            profit = (self.entry_price - price) * self.qty
            self.balance += (self.qty * self.entry_price + profit) * (1 - self.commission)

        self.qty           = 0.0
        self.position_side = 0
        self.entry_price   = 0.0

    def _update_net_worth(self, price):
        if self.position_side == 1:
            self.net_worth = self.balance + (self.qty * price)
        elif self.position_side == 2:
            profit = (self.entry_price - price) * self.qty
            self.net_worth = self.balance + (self.qty * self.entry_price + profit)
        else:
            self.net_worth = self.balance