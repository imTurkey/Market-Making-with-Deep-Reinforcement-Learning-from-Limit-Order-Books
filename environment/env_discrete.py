import numpy as np
import pandas as pd
import random
import math
import time
from datetime import timedelta

from .env_feature import EnvFeature
from .base_env import TRADE_UNIT

from utils import day2date, lob_norm

class EnvDiscrete(EnvFeature):
    """
    """
    def __init__(
            self, 
            code='000001', 
            day='20191101', 
            data_norm=True, 
            latency=1, 
            T=50, 
            # ablation states
            wo_lob_state=False,
            wo_market_state=False,
            wo_agent_state=False,
            # ablation rewards
            wo_dampened_pnl=False,
            wo_matched_pnl=False,
            wo_inv_punish=False,
            **kwargs
        ):
        super().__init__(**kwargs)
        print("Environment: EnvDiscrete")
        self.code = code
        self.day = day2date(day)

        self.latency = latency
        self.T = T

        # ablation
        self.wo_lob_state = wo_lob_state
        self.wo_market_state = wo_market_state
        self.wo_agent_state = wo_agent_state
        self.r_da = 0 if wo_dampened_pnl else 1
        self.r_ma = 0 if wo_matched_pnl else 1
        self.r_ip = 0 if wo_inv_punish else 1

        # Inventory punishment factor
        self.theta = 0.01
        self.eta = 0.5

        self.init_states()

        self.load_orderbook(code=code, day=day)
        self.load_price(code=code, day=day)
        self.load_trade(code=code, day=day)
        self.load_msg(code=code, day=day)

    def init_states(self):
        self.__states_space__ = dict()
        if not self.wo_lob_state:
            self.__states_space__['lob_state'] = dict(
                type='float',
                shape=(self.T,40,1)
                )
        if not self.wo_market_state:
            self.__states_space__['market_state'] = dict(
                type='float',
                shape=(24,)
                )
        if not self.wo_agent_state:
            self.__states_space__['agent_state'] = dict(
                type='float',
                shape=(24,)
                )
            
    def states(self):
        return self.__states_space__

    def actions(self):
        return dict(
                    type='int',
                    num_values=5
                )
    
    def max_episode_timesteps(self):
        return self.__max_episode_timesteps__

    def action2order(self, actions):
        # t-latency
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(self.i-self.latency)

        ask_price, bid_price = 0, 0
        ask_volume, bid_volume = -TRADE_UNIT,TRADE_UNIT

        if actions in range(7):
            # limit order
            if actions == 0:
                ask_price = t_1_a1_price
                bid_price = t_1_b1_price
            elif actions == 1:
                ask_price = t_1_a1_price
                bid_price = t_1_b1_price-0.01
            elif actions == 2: 
                ask_price = t_1_a1_price+0.01
                bid_price = t_1_b1_price
            elif actions == 3: 
                ask_price = t_1_a1_price+0.01
                bid_price = t_1_b1_price-0.01
            elif actions == 4: 
                ask_price = t_1_a1_price
                bid_price = t_1_b1_price-0.02
            elif actions == 5: 
                ask_price = t_1_a1_price+0.02
                bid_price = t_1_b1_price
            elif actions == 6: 
                ask_price = t_1_a1_price+0.02
                bid_price = t_1_b1_price-0.02

        elif actions==7:   
            # market order to clode position
            if self.inventory < 0:
                bid_price, bid_volume = np.inf, -self.inventory
            elif self.inventory > 0:
                ask_price, ask_volume = 0.01, -self.inventory
            else:
                trade_price, trade_volume = 0, 0

        # inventory limit
        if self.inventory < -10*TRADE_UNIT:
            ask_price=0
            ask_volume=0
        elif self.inventory > 10*TRADE_UNIT:
            bid_price=0
            bid_volume=0
        
        orders = {
            'ask_price': ask_price,
            'ask_vol': ask_volume,
            'bid_price': bid_price,
            'bid_vol': bid_volume
        }

        return orders

    def get_reward(self, trade_price, trade_volume):
        pnl = self.value - self.value_

        # Asymmetrically dampened PnL
        asymmetric_dampen = max(0, self.eta * pnl)
        dampened_pnl = pnl - asymmetric_dampen

        matched_pnl = (self.mid_price - trade_price) * trade_volume

        delta_inventory = abs(self.inventory) - abs(self.inventory_)
        # delta_inventory = max(0, delta_inventory)

        inventory_punishment = self.theta * (delta_inventory/TRADE_UNIT)
        # inventory_punishment = self.theta * (self.inventory/TRADE_UNIT)**2
        reward = pnl
        # reward = self.r_ma * matched_pnl + self.r_da * dampened_pnl - self.r_ip * inventory_punishment
        self.value_ = self.value

        return reward

    def get_state_at_t(self, t):
        self.__state__ = dict()

        if not self.wo_lob_state:
            lob = self.episode_state.iloc[t-self.T:t]
            mid_price = (lob.ask1_price + lob.bid1_price)/2
            lob_normed = lob_norm(lob, mid_price)
            self.__state__['lob_state'] = np.expand_dims(np.array(lob_normed), -1)

        if not self.wo_market_state:
            self.__state__['market_state'] = self._get_market_state(t) + self._get_order_strength_index(t)

        if not self.wo_agent_state:
            self.__state__['agent_state'] = [self.inventory/(10*TRADE_UNIT)]*12 + [t / self.episode_length]*12

        return self.__state__