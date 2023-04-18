from datetime import timedelta

from .base_env import BaseEnv

from utils import getRealizedVolatility, getRelativeStrengthIndex, getOrderStrengthIndex

class EnvFeature(BaseEnv):
    """
        Use this class to calculate your factor
    """
    def __init__(
            self, 
            **kwargs
        ):
        super().__init__(**kwargs)

    def _get_market_state(self,t):
        data_300s = self.price[(self.price.index<=self.episode_state.index[t])&(self.price.index>=self.episode_state.index[t]-timedelta(seconds=300))].midprice
        data_600s = self.price[(self.price.index<=self.episode_state.index[t])&(self.price.index>=self.episode_state.index[t]-timedelta(seconds=600))].midprice
        data_1800s = self.price[(self.price.index<=self.episode_state.index[t])&(self.price.index>=self.episode_state.index[t]-timedelta(seconds=1800))].midprice
        rv_300s = getRealizedVolatility(data_300s,resample='s')*1e4
        rv_600s = getRealizedVolatility(data_600s,resample='s')*1e4
        rv_1800s = getRealizedVolatility(data_1800s,resample='s')*1e4
        rsi_300s = getRelativeStrengthIndex(data_300s)
        rsi_600s = getRelativeStrengthIndex(data_600s)
        rsi_1800s = getRelativeStrengthIndex(data_1800s)
        return [rv_300s, rv_600s, rv_1800s, rsi_300s, rsi_600s, rsi_1800s]

    def _get_order_strength_index(self,t):
        data_10s = self.msg[(self.msg.index<=self.episode_state.index[t])&(self.msg.index>=self.episode_state.index[t]-timedelta(seconds=10))]
        data_60s = self.msg[(self.msg.index<=self.episode_state.index[t])&(self.msg.index>=self.episode_state.index[t]-timedelta(seconds=60))]
        data_300s = self.msg[(self.msg.index<=self.episode_state.index[t])&(self.msg.index>=self.episode_state.index[t]-timedelta(seconds=300))]

        svi_10s, sni_10s, lvi_10s, lni_10s, wvi_10s, wni_10s = getOrderStrengthIndex(data_10s)
        svi_60s, sni_60s, lvi_60s, lni_60s, wvi_60s, wni_60s = getOrderStrengthIndex(data_60s)
        svi_300s, sni_300s, lvi_300s, lni_300s, wvi_300s, wni_300s = getOrderStrengthIndex(data_300s)

        return [
            svi_10s, sni_10s, lvi_10s, lni_10s, wvi_10s, wni_10s,
            svi_60s, sni_60s, lvi_60s, lni_60s, wvi_60s, wni_60s,
            svi_300s, sni_300s, lvi_300s, lni_300s, wvi_300s, wni_300s
        ]