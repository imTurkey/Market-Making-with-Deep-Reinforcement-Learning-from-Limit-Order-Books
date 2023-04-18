import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm

# from tensorforce import Environment

from utils import day2date

TRADE_UNIT = 100

class BaseEnv():
    """
    """
    def __init__(
            self, 
            initial_value=0, 
            max_episode_timesteps=1000,
            data_dir='./data', 
            log=1, 
            experiment_name='', 
            **kwargs
            ):
        super().__init__()
        self.name = ''
        self.initial_value = initial_value
        self.__max_episode_timesteps__=max_episode_timesteps
        self.data_dir = data_dir
        self.log = log
        self.exp_name = experiment_name   

    '''
        You need to overload these functions
    '''   

    def states(self):
        raise NotImplementedError

    def actions(self):
        raise NotImplementedError
    
    def action2order(self):
        raise NotImplementedError
    
    def get_state_at_t(self, t):
        raise NotImplementedError  

    def get_reward(self, trade_price, trade_volume):
        # Define reward function here
        reward = self.value - self.value_
        self.value_ = self.value
        return reward

    '''
        Load data
    '''    

    def load_orderbook(self, code, day):
        ask = pd.read_csv(self.data_dir + f'/{code}/{day}/ask.csv')
        bid = pd.read_csv(self.data_dir + f'/{code}/{day}/bid.csv').drop(['timestamp'], axis = 1)

        self.orderbook = pd.concat([ask, bid], axis=1)
        self.orderbook.timestamp = pd.to_datetime(self.orderbook.timestamp)
        self.orderbook = self.orderbook[(f'{self.day} 09:30:00'<self.orderbook.timestamp)&(self.orderbook.timestamp<f'{self.day} 14:57:00')]
        self.orderbook = self.orderbook.set_index('timestamp')
        self.orderbook_length = len(self.orderbook)
        print('load lob done!', code, day)

    def load_orderqueue(self, code, day):
        pass

    def load_price(self, code, day):
        self.price = pd.read_csv(self.data_dir + f'/{code}/{day}/price.csv')
        self.price.timestamp = pd.to_datetime(self.price.timestamp)
        self.price = self.price.set_index('timestamp')
        self.price = self.price.loc[self.orderbook.index]

    def load_msg(self, code, day):
        self.msg = pd.read_csv(self.data_dir + f'/{code}/{day}/msg.csv')
        self.msg.timestamp = pd.to_datetime(self.msg.timestamp)
        self.msg = self.msg.set_index('timestamp')
        self.msg = self.msg.loc[self.orderbook.index]
    
    def load_order(self, code, day):
        order_columns = pd.read_csv('raw/GTA_SZL2_ORDER.csv')
        self.order = pd.read_csv(f'raw/SZL2_ORDER_{code}_{day[:6]}.csv', names=list(order_columns), low_memory=False)
        self.order.TradingTime = pd.to_datetime(self.order.TradingTime)
        self.order = self.order[self.order.TradingDate==int(day)]
        self.order = self.order[(f'{self.day} 09:30:00'<self.order.TradingTime)&(self.order.TradingTime<f'{self.day} 14:57:00')]

    def load_trade(self, code, day):
        trade_columns = pd.read_csv('raw/GTA_SZL2_TRADE.csv')
        self.trade = pd.read_csv(f'raw/SZL2_TRADE_{code}_{day[:6]}.csv', names=list(trade_columns))
        self.trade.TradingTime = pd.to_datetime(self.trade.TradingTime)
        self.trade = self.trade[self.trade.TradingDate==int(day)]
        self.trade = self.trade[self.trade.TradeType=="F"]
        self.trade = self.trade[(f'{self.day} 09:30:00'<self.trade.TradingTime)&(self.trade.TradingTime<f'{self.day} 14:57:00')]
        
        self.is_trade = pd.DataFrame(index=self.orderbook.index,columns=['is_trade'])
        self.is_trade['is_trade'] = 0
        self.is_trade.loc[set(self.trade.TradingTime)] = 1

    '''
        Common function
    '''

    def reset_seq(self, timesteps_per_episode=None, episode_idx=None):
        self.episode_idx = episode_idx
        if timesteps_per_episode == None:
            self.episode_start = 0
            self.episode_end = len(self.orderbook)
            self.episode_state = self.orderbook
        else:
            self.episode_start = timesteps_per_episode * episode_idx
            self.episode_end = min(self.episode_start + timesteps_per_episode, len(self.orderbook))
            self.episode_state = self.orderbook.iloc[self.episode_start:self.episode_end]
        
        self.episode_length = len(self.episode_state)
        
        episode_is_trade = self.is_trade.iloc[self.episode_start:self.episode_end]
        has_trade_index = np.where(episode_is_trade==1)[0]
        has_trade_index = has_trade_index[has_trade_index>self.T]
        self.index_iterator = iter(has_trade_index)

        self.cash = self.value_ = self.value = self.initial_value
        self.holding_pnl_total = self.trading_pnl_total = 0
        self.inventory = 0
        self.volume = 0
        self.episode_reward = 0
        self.mid_price_ = None
        self.action_his = []
        self.reward_dampened_pnl = 0
        self.reward_trading_pnl = 0
        self.reward_inventory_punishment = 0
        self.reward_spread_punishment = 0

        # log for trade
        self.logger = self.price.iloc[self.episode_start:self.episode_end].copy()
        columns=['ask_price', 'bid_price', 'trade_price', 'trade_volume', 'value', 'volume', 'cash', 'inventory']
        for column in columns:
            self.logger[column] = np.nan

        self.i = next(self.index_iterator)
        self.i_ = next(self.index_iterator)
        state = self.get_state_at_t(self.i-self.latency)

        if self.log >= 1:
            print(f'Reset env {self.name} {self.code}, {self.day}, from {self.episode_state.index[0]} to {self.episode_state.index[-1]}')
            self.pbar = tqdm(total=self.episode_length)
            self.pbar.update(self.i)

        return state
    
    def reset_random(self, timesteps_per_episode=2000):
        self.episode_start = np.random.randint(0, len(self.orderbook) - timesteps_per_episode)
        self.episode_end = min(self.episode_start + timesteps_per_episode, len(self.orderbook))
        self.episode_state = self.orderbook.iloc[self.episode_start:self.episode_end]
        
        self.episode_length = len(self.episode_state)
        
        episode_is_trade = self.is_trade.iloc[self.episode_start:self.episode_end]
        has_trade_index = np.where(episode_is_trade==1)[0]
        has_trade_index = has_trade_index[has_trade_index>self.T]
        self.index_iterator = iter(has_trade_index)

        self.cash = self.value_ = self.value = self.initial_value
        self.holding_pnl_total = self.trading_pnl_total = 0
        self.inventory = 0
        self.volume = 0
        self.episode_reward = 0
        self.mid_price_ = None
        self.action_his = []
        self.reward_dampened_pnl = 0
        self.reward_trading_pnl = 0
        self.reward_inventory_punishment = 0
        self.reward_spread_punishment = 0

        # log for trade
        self.logger = self.price.iloc[self.episode_start:self.episode_end].copy()
        columns=['ask_price', 'bid_price', 'trade_price', 'trade_volume', 'value', 'volume', 'cash', 'inventory']
        for column in columns:
            self.logger[column] = np.nan

        self.i = next(self.index_iterator)
        self.i_ = next(self.index_iterator)
        state = self.get_state_at_t(self.i-self.latency)

        if self.log:
            print(f'Reset env {self.name} {self.code}, {self.day}, from {self.episode_state.index[0]} to {self.episode_state.index[-1]}')
            self.pbar = tqdm(total=self.episode_length)
            self.pbar.update(self.i)

        return state
    
    def execute(self, actions):
        self.action_his.append(actions)
        # t
        self.mid_price, self.ask1_price, self.bid1_price, self.lob_spread = self.get_price_info(self.i)
        if self.mid_price_ == None:
            self.mid_price_ = self.mid_price

        orders = self.action2order(actions)
        # inventory limit
        if self.inventory < -10*TRADE_UNIT:
            orders['ask_price']=0
        elif self.inventory > 10*TRADE_UNIT:
            orders['bid_price']=0

        trade_price, trade_volume = self.match(orders)

        self.update_agent(trade_price, trade_volume)

        # log for trade result
        self.logger.iloc[self.i, -8:] = [orders['ask_price'], orders['bid_price'], trade_price, trade_volume, self.value, self.volume, self.cash, self.inventory]

        # if trade_volume:
        #     print(self.i, 'ask1:', self.ask1_price, 'bid1:', self.bid1_price, 'buy' if trade_volume>0 else 'sell', 'at', trade_price)
        if self.log >= 1:
            self.pbar.update(self.i_ - self.i)
        
        self.i = self.i_
        # Termination conditions
        terminal = False
        try:
            self.i_ = next(self.index_iterator)
        except:
            terminal = True

        reward = self.get_reward(trade_price, trade_volume)
        self.mid_price_ = self.mid_price

        # close position
        if terminal:
            trade_price, trade_volume = self.close_position()
            reward += self.get_reward(trade_price, trade_volume)

        self.episode_reward += reward

        # log for result
        if terminal:
            self.post_experiment()

        state = self.get_state_at_t(self.i-self.latency)
        
        return state, terminal, reward

    def match(self, actions):
        trade_volume = 0
        trade_price = 0
        ask_price, ask_volume, bid_price, bid_volume = actions.values()

        # trade
        now_t = self.trade[self.trade.TradingTime==self.episode_state.index[self.i]]
        now_trading_price_max = now_t.TradePrice.max()
        now_trading_price_max_v = now_t[now_t.TradePrice==now_trading_price_max].TradeVolume.sum()
        now_trading_price_min = now_t.TradePrice.min()
        now_trading_price_min_v = now_t[now_t.TradePrice==now_trading_price_min].TradeVolume.sum()

        # t - 1
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(self.i-1)

        # sell order
        if ask_price and ask_volume:
            if ask_price <= t_1_b1_price:
                # market order
                trade_price, trade_volume = t_1_b1_price, ask_volume
                # print("market order sell at", trade_price)
            else:
                # limit order
                if now_trading_price_max > ask_price:
                    # all deal
                    trade_price, trade_volume = ask_price, ask_volume
                    # print("limit order sell at", trade_price)

                # we assume that our quotes rest at the back of the queue 
                elif now_trading_price_max == ask_price:
                    # deal probability: traded volume/all volume in this level
                    lob_depth = self.episode_state.iloc[self.i].ask1_volume
                    transac_prob = now_trading_price_max_v/(now_trading_price_max_v+lob_depth)
                    is_transac = np.random.choice([1, 0], p=[transac_prob, 1-transac_prob])
                    if is_transac:
                        trade_price, trade_volume = ask_price, ask_volume

        # buy order
        if bid_price and bid_volume:
            if bid_price >= t_1_a1_price:
                # market order
                trade_price, trade_volume = t_1_a1_price, bid_volume
                # print("market order buy at", trade_price)
            else:
                if now_trading_price_min < bid_price:
                    trade_price, trade_volume = bid_price, bid_volume
                    # print("limit order buy at", trade_price)

                # we assume that our quotes rest at the back of the queue
                elif now_trading_price_min == bid_price:
                    lob_depth = self.episode_state.iloc[self.i].bid1_volume
                    transac_prob = now_trading_price_min_v/(now_trading_price_min_v+lob_depth)
                    is_transac = np.random.choice([1, 0], p=[transac_prob, 1-transac_prob])
                    if is_transac:
                        trade_price, trade_volume = bid_price, bid_volume

        return trade_price, trade_volume
    
    def close_position(self):
        # t - 1
        t_1_mid_price, t_1_a1_price, t_1_b1_price, t_1_spread = self.get_price_info(self.i-1)

        # Market order
        if self.inventory < 0:
            # Buy
            trade_price, trade_volume = t_1_a1_price, -self.inventory
            self.volume += trade_volume
        elif self.inventory > 0:
            # Sell
            trade_price, trade_volume = t_1_b1_price, -self.inventory
        else:
            trade_price, trade_volume = 0, 0

        self.update_agent(trade_price, trade_volume)

        # log for trade result
        self.logger.iloc[self.i, -6:] = [trade_price, trade_volume, self.value, self.volume, self.cash, self.inventory]

        return trade_price, trade_volume
    
    def update_agent(self, trade_price, trade_volume):
        self.inventory_ = self.inventory
        self.inventory += trade_volume
        self.cash -= trade_volume*trade_price 
        self.value = self.get_value(self.mid_price)

        volume = max(0, trade_volume*trade_price) # only count for buy
        self.volume += volume

    def get_price_info(self, i):
        price = self.price[self.price.index==self.episode_state.index[i]]

        bid1_price = price.bid1_price.item()
        ask1_price = price.ask1_price.item()
        bid1_price, ask1_price = round(bid1_price,2), round(ask1_price,2)
        mid_price = (bid1_price+ask1_price)/2
        spread = ask1_price - bid1_price

        return mid_price, ask1_price, bid1_price, spread

    def get_value(self, price):
        return self.cash + self.inventory*price

    '''
        For evaluation and save trading log
    '''

    def post_experiment(self, save=False):
        logger_wo_exit_market = self.logger[(self.logger.ask_price != 0) & (self.logger.bid_price != 0)]
        self.episode_avg_spread = (logger_wo_exit_market.ask_price - logger_wo_exit_market.bid_price).mean()
        self.episode_avg_position = self.logger.inventory.mean()
        self.episode_avg_abs_position = self.logger.inventory.abs().mean()
        self.episode_profit_ratio = self.value/(self.volume+1e-7)
        self.pnl = self.value - self.initial_value
        self.nd_pnl = self.pnl/self.episode_avg_spread
        self.pnl_map = self.pnl/(self.episode_avg_abs_position+1e-7)

        if self.log >= 1:
            print(
                "PnL:", self.pnl, 
                "Holding PnL", self.holding_pnl_total,
                "Trading PnL", self.trading_pnl_total,
                "ND-PnL:", self.nd_pnl,
                "PnL-MAP:", self.pnl_map,
                "Trading volume:", self.volume, 
                "Profit ratio:", self.episode_profit_ratio, 
                "Averaged position:",self.episode_avg_position, 
                "Averaged Abs position:",self.episode_avg_abs_position, 
                "Averaged spread:", self.episode_avg_spread,
                "Episodic reward:", self.episode_reward
                )
            self.pbar.close()

        if self.log >= 2:
            trade_log = self.logger[(self.logger.trade_volume > 0)|(self.logger.trade_volume < 0)]
            for i in range(len(trade_log)):
                item = trade_log.iloc[i]
                if item.trade_volume > 0:
                    print(item.name, 'BUY at', item.trade_price, 'inventory', item.inventory, 'value', item.value)
                elif item.trade_volume < 0:
                    print(item.name, 'SELL at', item.trade_price, 'inventory', item.inventory, 'value', item.value)

        if save:
            now_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
            log_file = f"./log/{self.exp_name}_{self.code}_{self.day}_{now_time}.csv"
            self.logger.to_csv(log_file)
            print("Trading log saved to", log_file)

    def get_final_result(self):
        return dict(
            pnl=self.pnl, 
            nd_pnl=self.nd_pnl, 
            pnl_map=self.pnl_map,
            profit_ratio=self.episode_profit_ratio, 
            avg_position=self.episode_avg_position, 
            avg_abs_position=self.episode_avg_abs_position, 
            avg_spread=self.episode_avg_spread,
            volume=self.volume, 
            episode_reward=self.episode_reward
        )