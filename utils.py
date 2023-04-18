import pandas as pd
import numpy as np
import math

def min_max_norm(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    if ranges==0:
        return data*0
    normData = data - minVals
    normData = normData/ranges
    return normData

def z_norm(data):
    return (data-data.mean())/(data.std()+1e-7)

def lob_norm(data_, midprice):
    data = data_.copy()
    for i in range(10):
        data[f'ask{i+1}_price'] = data[f'ask{i+1}_price']/(midprice+1e-7) - 1
        data[f'bid{i+1}_price'] = data[f'bid{i+1}_price']/(midprice+1e-7) - 1
        # data[f'ask{i+1}_price'] = z_norm(data[f'ask{i+1}_price'])
        # data[f'bid{i+1}_price'] = z_norm(data[f'bid{i+1}_price'])
        data[f'ask{i+1}_volume'] = data[f'ask{i+1}_volume']/data[f'ask{i+1}_volume'].max()
        data[f'bid{i+1}_volume'] = data[f'bid{i+1}_volume']/data[f'bid{i+1}_volume'].max()

    return data

def onehot_label(targets):
    from tensorflow import keras
    # targets: pd.DataFrame len(data)*n_horizons
    all_label = []
    for i in range(targets.shape[1]):
        label = targets.iloc[:,i] - 1
        label = keras.utils.to_categorical(label, 3)
        # label = label.reshape(len(label), 1, 3)
        all_label.append(label)
    return np.hstack(all_label)

def day2date(day):
    day = list(day)
    day.insert(4,'-')
    day.insert(7,'-')
    date = ''.join(day)
    return date

def pd_is_equal(state_1, state_2):
    tmp_1 = state_1.iloc[:,1:]
    tmp_2 = state_2.iloc[:,1:]
    return tmp_1.equals(tmp_2)

def load_data(code, datelist, horizon=10):
    if type(datelist) is str:
        datelist = [datelist]
    data_list = []
    for day in datelist:
        ask = pd.read_csv(f"data/{code}/{day}/ask.csv")
        bid = pd.read_csv(f"data/{code}/{day}/bid.csv").drop(['timestamp'], axis = 1)
        price = pd.read_csv(f"data/{code}/{day}/price.csv").drop(['timestamp', 'ask1_price', 'bid1_price'], axis = 1)
        data = pd.concat([ask, bid, price], axis=1)
        data['date'] = data['timestamp'].str.split(expand=True)[0]
        data['time'] = data['timestamp'].str.split(expand=True)[1]
        data.drop('timestamp', axis=1, inplace=True)

        data['y']=getLabel(data.midprice, horizon)

        data_list.append(data)
    return pd.concat(data_list)

def getLabel(mid_price, horizon, threshold=1e-5):
    price_past = mid_price.rolling(window=horizon).mean()

    price_future = mid_price.copy()
    price_future[:-horizon] = price_past[horizon:]
    price_future[-horizon:] = np.nan

    pct_change = (price_future - price_past)/price_past
    pct_change[pct_change>=threshold] = 1
    pct_change[(pct_change<threshold) & (-threshold<pct_change)] = 2
    pct_change[pct_change<=-threshold] = 3
    return pct_change

def process_data(data):
    data = data[(data.time > '10:00:00')&(data.time < '14:30:00')]
    data = data.dropna()
    data.y = data.y.astype(int)

    for i in range(10):
        data[f'ask{i+1}_price'] = data[f'ask{i+1}_price']/data['midprice'] - 1
        data[f'bid{i+1}_price'] = data[f'bid{i+1}_price']/data['midprice'] - 1
        # data[f'ask{i+1}_price'] = z_norm(data[f'ask{i+1}_price'])
        # data[f'bid{i+1}_price'] = z_norm(data[f'bid{i+1}_price'])
        data[f'ask{i+1}_volume'] = data[f'ask{i+1}_volume']/data[f'ask{i+1}_volume'].max()
        data[f'bid{i+1}_volume'] = data[f'bid{i+1}_volume']/data[f'bid{i+1}_volume'].max()

    return data.set_index(['date', 'time'])

def reorder(data):
    '''
    reorder the data to this order:
    ask1_v, ask1_p, bid1_v, bid1_p ... ask10_v, ask10_p, bid10_v, bid10_p
    '''
    data=np.array(data)
    data=data.reshape(data.shape[0], 4, 10)
    data= np.transpose(data, (0,2,1))
    data = data.reshape(data.shape[0], -1)
    return data

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY

def price_legal_check(ask_price, bid_price):
    # legal check
    ask_price = math.ceil(100*ask_price)/100
    bid_price = math.floor(100*bid_price)/100
    return ask_price, bid_price

def getRealizedVolatility(data, resample='min'):
    if resample:
        data = data.resample(resample).last()
    
    midprice_lag = data.shift(1)
    midprice_log = data.apply(np.log)
    midprice_lag_log = midprice_lag.apply(np.log)
    r = midprice_log - midprice_lag_log
    r2 = r*r
    rv = r2.sum()

    return rv

def getRelativeStrengthIndex(data):
    length = len(data)
    data = data.resample('s').last()
    data = data.pct_change(1)
    gain = data[data>0].sum()/length
    loss = -data[data<0].sum()/length
    if gain or loss:
        rsi = gain/(gain+loss)
    else:
        rsi = .5
    return rsi

def getOrderStrengthIndex(data):
    '''
    data: msg
    columns:[market_buy_volume  market_buy_n  market_sell_volume  market_sell_n  limit_buy_volume  limit_buy_n  limit_sell_volume  limit_sell_n  withdraw_buy_volume  withdraw_buy_n  withdraw_sell_volume  withdraw_sell_n]
    '''
    market_volume_intensity = (data.market_buy_volume.sum() - data.market_sell_volume.sum())/(data.market_buy_volume.sum() + data.market_sell_volume.sum() + 1e-7)
    market_number_intensity = (data.market_buy_n.sum() - data.market_sell_n.sum())/(data.market_buy_n.sum() + data.market_sell_n.sum() + 1e-7)
    limit_volume_intensity = (data.limit_buy_volume.sum() - data.limit_sell_volume.sum())/(data.limit_buy_volume.sum() + data.limit_sell_volume.sum() + 1e-7)
    limit_number_intensity = (data.limit_buy_n.sum() - data.limit_sell_n.sum())/(data.limit_buy_n.sum() + data.limit_sell_n.sum() + 1e-7)
    withdraw_volume_intensity = (data.withdraw_buy_volume.sum() - data.withdraw_sell_volume.sum())/(data.withdraw_buy_volume.sum() + data.withdraw_sell_volume.sum() + 1e-7)
    withdraw_number_intensity = (data.withdraw_buy_n.sum() - data.withdraw_sell_n.sum())/(data.withdraw_buy_n.sum() + data.withdraw_sell_n.sum() + 1e-7)

    return market_volume_intensity, market_number_intensity, limit_volume_intensity, limit_number_intensity, withdraw_volume_intensity, withdraw_number_intensity