
import time
import ccxt
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime
import requests
from datetime import datetime, timedelta
import sys

from pprint import pprint
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')


class bt():

    '''

    '''

    def __init__(self, symbol=None,  data=None, resolution=None, delay=None, model=None, window=None, threshold=None, long_or_short=None, direction=None, tc=None, max_days_in_position=None):

        self.symbol = symbol
        self.resolution = resolution
        self.delay = delay
        self.data = data
        self.model = model
        self.window = window
        self.threshold = threshold
        self.long_or_short = long_or_short
        self.direction = direction
        self.tc = tc
        self.max_days_in_position = max_days_in_position

    def get_current_parmas(self, resolution=None, delay=None, data=None, model=None, window=None, threshold=None, long_or_short=None, direction=None, tc=None, max_days_in_position=None):

        self.params = {
            'symbol': self.symbol,
            'resolution': self.resolution,
            'delay': self.delay,
            'data': self.data,
            'model': self.model,
            'window': self.window,
            'threshold': self.threhold,
            'long_or_short': self.long_or_short,
            'direction': self.direction,
            'trancost': self.tc,
            'max_days_in_position': self.max_days_in_position
        }

        return self.params

    def run_strategy(self, resolution=None, delay=None, data=None, model=None, window=None, threshold=None, long_or_short=None, direction=None, tc=None, max_days_in_position=None):
        '''

        data            : price , factor ad index in datetime
        resolution      : 1min, 5min, 10min, 1hour, 2hour, 4hour, daily 
        model           : z_score, mad_z_score, ma_x, ma_diff, percentile, min_max
        window          : 1st parameters
        threshold       : 2nd parameters
        long_or_short   : long short, long, short
        direction:      : momentum, reversion
        trancost        : percentage

        '''

        if resolution is not None:
            self.resolution = resolution
        if delay is not None:
            self.delay = delay
        if data is not None:
            self.data = data
        if model is not None:
            self.model = model
        if window is not None:
            self.window = window
        if threshold is not None:
            self.threshold = threshold
        if long_or_short is not None:
            self.long_or_short = long_or_short
        if direction is not None:
            self.direction = direction
        if tc is not None:
            self.tc = tc

        # Set the annualization factor based on the resolution
        if self.resolution == '1min':
            annual_factor = 252 * 390
        elif self.resolution == '5min':
            annual_factor = 252 * 78
        elif self.resolution == '10min':
            annual_factor = 252 * 39
        elif self.resolution == '1hour':
            annual_factor = 252 * 6.5
        elif self.resolution == '2hour':
            annual_factor = 252 * 3.25
        elif self.resolution == '4hour':
            annual_factor = 252 * 1.625
        elif self.resolution == 'daily':
            annual_factor = 252
        else:
            raise ValueError("Invalid resolution provided")

        df = data.copy()

        df['log_return'] = np.log(df['price'] / df['price'].shift(1))

        def z_score(df, window, threshold, long_or_short):

            df['ma'] = df['factor'].rolling(window).mean()
            df['sd'] = df['factor'].rolling(window).std()
            # zscore is n standard d away from mean
            df['z'] = (df['factor'] - df['ma']) / df['sd']

            if self.direction == 'momentum':
                if long_or_short == 'long short':
                    df['pos'] = np.where(df['z'] > threshold,
                                         1, np.where(df['z'] < -threshold, -1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['z'] > threshold,
                                         1, np.where(df['z'] < -threshold, 0, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(df['z'] > threshold,
                                         0, np.where(df['z'] < -threshold, -1, 0))
            else:
                if long_or_short == 'long short':
                    df['pos'] = np.where(
                        df['z'] > threshold, -1, np.where(df['z'] < -threshold, 1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['z'] > threshold,
                                         0, np.where(df['z'] < -threshold, 1, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(
                        df['z'] > threshold, -1, np.where(df['z'] < -threshold, 0, 0))
            return df['pos']

        def mad_z_score(df, window, threshold, long_or_short, direction):

            df['median'] = df['factor'].rolling(window).median()
            df['mad'] = df['factor'].rolling(window).apply(
                lambda x: np.median(np.abs(x - np.median(x))))

            df['z_mad'] = (df['factor'] - df['median']) / (df['mad'] * 1.4826)

            # Generate positions based on the Z-score
            if self.direction == 'momentum':
                if long_or_short == 'long short':
                    df['pos'] = np.where(df['z_mad'] > threshold, 1, np.where(
                        df['z_mad'] < -threshold, -1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['z_mad'] > threshold, 1, np.where(
                        df['z_mad'] < -threshold, 0, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(df['z_mad'] > threshold, 0, np.where(
                        df['z_mad'] < -threshold, -1, 0))
            else:
                if long_or_short == 'long short':
                    df['pos'] = np.where(
                        df['z_mad'] > threshold, -1, np.where(df['z_mad'] < -threshold, 1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['z_mad'] > threshold, 0, np.where(
                        df['z_mad'] < -threshold, 1, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(
                        df['z_mad'] > threshold, -1, np.where(df['z_mad'] < -threshold, 0, 0))

            return df['pos']

        def ma_x(df, window, threshold, long_or_short):

            df['sma_1'] = df['factor'].rolling(window).mean()
            df['sma_2'] = df['factor'].rolling(threshold).mean()

            if self.direction == 'momentum':
                if long_or_short == 'long short':
                    df['pos'] = np.where(df['sma_1'] > df['sma_2'], 1, np.where(
                        df['sma_1'] < df['sma_2'], -1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['sma_1'] > df['sma_2'], 1, np.where(
                        df['sma_1'] < df['sma_2'], 0, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(df['sma_1'] > df['sma_2'], 0, np.where(
                        df['sma_1'] < df['sma_2'], -1, 0))
            else:
                if long_or_short == 'long short':
                    df['pos'] = np.where(
                        df['sma_1'] > df['sma_2'], -1, np.where(df['sma_1'] < df['sma_2'], 1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['sma_1'] > df['sma_2'], 0, np.where(
                        df['sma_1'] < df['sma_2'], 1, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(
                        df['sma_1'] > df['sma_2'], -1, np.where(df['sma_1'] < df['sma_2'], 0, 0))
            return df['pos']

        def ma_diff(df, window, threshold, long_or_short):

            df['ma'] = df['factor'].rolling(window).mean()
            df['ma_diff'] = (df['factor'] / df['ma']) - 1.0

            if self.direction == 'momentum':
                if long_or_short == 'long short':
                    df['pos'] = np.where(df['ma_diff'] > threshold, 1, np.where(
                        df['ma_diff'] < -threshold, -1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['ma_diff'] > threshold, 1, np.where(
                        df['ma_diff'] < -threshold, 0, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(df['ma_diff'] > threshold, 0, np.where(
                        df['ma_diff'] < -threshold, -1, 0))
            else:
                if long_or_short == 'long short':
                    df['pos'] = np.where(
                        df['ma_diff'] > threshold, -1, np.where(df['ma_diff'] < -threshold, 1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['ma_diff'] > threshold, 0, np.where(
                        df['ma_diff'] < -threshold, 1, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(
                        df['ma_diff'] > threshold, -1, np.where(df['ma_diff'] < -threshold, 0, 0))
            return df['pos']

        def percentile(df, window, threshold, long_or_short):

            df['percentile_high'] = df['factor'].rolling(window).quantile(
                1 - threshold)  # threhold 0.05 = 95 prcentile
            df['percentile_low'] = df['factor'].rolling(window).quantile(
                threshold)  # threhold 0.95 = 5 prcentile

            if self.direction == 'momentum':
                if long_or_short == 'long short':
                    df['pos'] = np.where(df['factor'] > df['percentile_high'], 1, np.where(
                        df['factor'] < df['percentile_low'], -1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['factor'] > df['percentile_high'], 1, np.where(
                        df['factor'] < df['percentile_low'], 0, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(df['factor'] > df['percentile_high'], 0, np.where(
                        df['factor'] < df['percentile_low'], -1, 0))
            else:
                if long_or_short == 'long short':
                    df['pos'] = np.where(df['factor'] > df['percentile_high'], -1,
                                         np.where(df['factor'] < df['percentile_low'], 1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['factor'] > df['percentile_high'], 0, np.where(
                        df['factor'] < df['percentile_low'], 1, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(df['factor'] > df['percentile_high'], -1,
                                         np.where(df['factor'] < df['percentile_low'], 0, 0))

            return df['pos']

        # why needs 1 - Threshold >>> less intutitive
        def min_max(df, window, threshold, long_or_short):

            df['min'] = df['factor'].rolling(window).min()
            df['max'] = df['factor'].rolling(window).max()
            df['x'] = (df['factor'] - df['min']) / (df['max'] - df['min'])

            # threhold 0.05 = 95
            # threhold 0.95 = 05

            if self.direction == 'momentum':
                if long_or_short == 'long short':
                    df['pos'] = np.where(df['x'] > (
                        1 - threshold), 1, np.where(df['x'] < (threshold), -1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['x'] > (
                        1 - threshold), 1, np.where(df['x'] < (threshold), 0, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(df['x'] > (
                        1 - threshold), 0, np.where(df['x'] < (threshold), -1, 0))
            else:
                if long_or_short == 'long short':
                    df['pos'] = np.where(df['x'] > (
                        1 - threshold), -1, np.where(df['x'] < (threshold), 1, 0))
                elif long_or_short == 'long':
                    df['pos'] = np.where(df['x'] > (
                        1 - threshold), 0, np.where(df['x'] < (threshold), 1, 0))
                elif long_or_short == 'short':
                    df['pos'] = np.where(df['x'] > (
                        1 - threshold), -1, np.where(df['x'] < (threshold), 0, 0))
            return df['pos']

        if model == 'z_score':
            df['pos'] = z_score(df, window, threshold, long_or_short)
        elif model == 'mad_z_score':
            df['pos'] = mad_z_score(df, window, threshold, long_or_short)
        elif model == 'ma_x':
            df['pos'] = ma_x(df, window, threshold, long_or_short)
        elif model == 'ma_diff':
            df['pos'] = ma_diff(df, window, threshold, long_or_short)
        elif model == 'min_max':
            df['pos'] = min_max(df, window, threshold, long_or_short)
        elif model == 'percentile':
            df['pos'] = percentile(df, window, threshold, long_or_short)
        else:
            raise ValueError("Invalid model provided")

        df['pos_delay'] = df['pos'].shift(delay).fillna(0)
        df['pos_delay_t-1'] = df['pos_delay'].shift(1).fillna(0)
        df['trade'] = df['pos_delay'] - df['pos_delay_t-1']
        df['cost'] = df['trade'].abs() * df['price'] * self.tc/100
        df['strategy_return'] = df['pos_delay_t-1'] * \
            df['log_return'] - df['cost']

        if max_days_in_position is not None:
            df['days_in_position'] = df['pos_delay'].groupby(
                (df['pos_delay'] != df['pos_delay'].shift()).cumsum()).cumcount()
            df['pos_delay'] = np.where(
                df['days_in_position'] >= max_days_in_position, 0, df['pos_delay'])

        df.dropna(inplace=True)
        df['cum_return'] = df['log_return'].cumsum().apply(np.exp) - 1
        df['cum_strategy'] = df['strategy_return'].cumsum().apply(np.exp) - 1

        self.result = df

        return self.result

    def get_metrics(self):
        if self.result is None:
            print('No result to plot yet. Run a strategy first!')

        else:
            df = self.result
            bnh_return = df['cum_return'].iloc[-1]
            finalReturn = df['cum_strategy'].iloc[-1]
            bnh_return_APY = (df['log_return'].apply(
                np.exp)-1).mean() * self.annual_factor
            APY = (df['strategy_return'].apply(
                np.exp)-1).mean() * self.annual_factor
            risk = (df['strategy_return'].apply(np.exp) -
                    1).std() * (self.annual_factor ** 0.5)
            sharpeRatio = APY / risk
            drawDown = df['cum_strategy'] / df['cum_strategy'].cummax() - 1
            maxDrawDown = (-drawDown).max()
            calmarRatio = APY / maxDrawDown
            no_trade = df['trade'].abs().sum()
            no_days_in_position = df['pos_delay_t-1'].abs().sum()
            strategyDrawDown = drawDown
            strategyDrawDown = strategyDrawDown[strategyDrawDown == 0]
            if len(strategyDrawDown) > 1:
                drawDownPeriods = (strategyDrawDown.index[1:].to_pydatetime(
                ) - strategyDrawDown.index[:-1].to_pydatetime()).max()
            else:
                drawDownPeriods = None

            metrics_dict = {
                'window': self.window,
                'threshold': self.threshold,
                'finalReturn': finalReturn,
                'APY': APY,
                'risk': risk,
                'sharpeRatio': sharpeRatio,
                'maxDrawDown': maxDrawDown,
                'calmarRatio': calmarRatio,
                'drawDownPeriods': drawDownPeriods,
                '# of Trade': no_trade,
                '# of days in position': no_days_in_position,
                'bnh Return APY': bnh_return_APY,
                'outperform': APY - bnh_return_APY
            }
            print(metrics_dict)

            # return self.result, pd.Series([window, threshold, finalReturn, APY, risk, sharpeRatio, maxDrawDown, calmarRatio, drawDownPeriods, bnh_return_APY, APY - bnh_return_APY],
            #                               index=['window', 'threshold', 'finalReturn', 'APY', 'risk', 'sharpeRatio', 'maxDrawDown', 'calmarRatio', 'drawDownPeriods', 'bnh Return', 'outperform'])

            return metrics_dict

    def plot_strategy(self):
        if self.result is None:
            print('No result to plot yet. Run a strategy first!')
        else:
            self.result[['cum_return', 'cum_strategy']].plot(figsize=(
                15, 7), title=f'{self.symbol} Equity Curve, window: {self.window}, threshold: {self.threshold}')

    def optimize_para(self, range1, range2):
        self.opt = []
        if self.result is None:
            print('No result to process yet. Run a strategy first!')
        else:
            for (sma1, sma2) in [(sma1, sma2) for sma1 in range1 for sma2 in range2]:
                self.opt.append(self.update_and_run((sma1, sma2)))

            self.opt_final = pd.DataFrame(self.opt).sort_values(
                by=['sharpeRatio'], ascending=False)
            return self.opt_final
