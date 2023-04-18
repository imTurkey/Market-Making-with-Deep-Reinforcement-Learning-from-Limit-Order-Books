import numpy as np
import pandas as pd
import random
import math
import time

from .env_feature import EnvFeature
from .base_env import TRADE_UNIT

from utils import day2date, lob_norm, price_legal_check

class EnvContinuous(EnvFeature):
    """
    """
    def __init__(
            self, 
            code='000001', 
            day='20191101', 
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
        self.name = "Continuous"
        print("Environment:", self.name)
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
        self.eta = 0.9

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
                    type='float',
                    shape=(2,),
                    min_value=-1,
                    max_value=1
                )

    def max_episode_timesteps(self):
        return self.__max_episode_timesteps__

    def action2order(self, actions):
        # t-latency
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(self.i-self.latency)

        # action 1
        # actions in [0, 1]
        delta_price = actions[0]*0.05
        spread = actions[1]*0.1
        if self.inventory > 0:
            reservation = t_1_mid_price - delta_price
        elif self.inventory < 0:
            reservation = t_1_mid_price + delta_price
        else:
            reservation = t_1_mid_price
        ask_price = reservation + spread/2
        bid_price = reservation - spread/2

        # action 2
        # actions in [-1, 1]
        # delta_price = actions[0]*0.05
        # spread = abs(actions[1])*0.1
        # reservation = t_1_mid_price - delta_price
        # ask_price = reservation + spread/2
        # bid_price = reservation - spread/2

        # action 3
        # actions in [0, 1]
        # ask_price = t_1_a1_price + actions[0]*0.1
        # bid_price = t_1_b1_price - actions[1]*0.1
        # reservation = (ask_price + bid_price)/2
        # spread = ask_price - bid_price

        ask_price, bid_price = price_legal_check(ask_price, bid_price)

        # save for log
        self.reservation = reservation
        self.spread = spread
        
        orders = {
            'ask_price': ask_price,
            'ask_vol': -TRADE_UNIT,
            'bid_price': bid_price,
            'bid_vol': TRADE_UNIT
        }
        return orders

    def get_reward(self, trade_price, trade_volume):
        pnl = self.value - self.value_

        # Asymmetrically dampened PnL
        asymmetric_dampen = max(0, self.eta * pnl)
        dampened_pnl = pnl - asymmetric_dampen

        matched_pnl = (self.mid_price - trade_price) * trade_volume

        # delta_inventory = abs(self.inventory) - abs(self.inventory_)
        # delta_inventory = max(0, delta_inventory)
        # inventory_punishment = self.theta * (delta_inventory/TRADE_UNIT)

        inventory_punishment = self.theta * (self.inventory/TRADE_UNIT)**2

        # spread punishment
        if self.inventory:
            spread_punishment = 0
        else:
            spread_punishment = 100*self.spread if self.spread > 0.02 else 0
        
        reward = pnl - spread_punishment#self.r_ma * matched_pnl + self.r_da * dampened_pnl - self.r_ip * inventory_punishment - spread_punishment
        
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