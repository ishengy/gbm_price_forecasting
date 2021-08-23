# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 04:03:59 2021

@author: Ivan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from scipy.stats import norm
import statsmodels.api as sm

git_path = 'D:/Documents/Github'
os.chdir(git_path + '/gbm_stock_prediction')
#os.chdir('/Users/isheng/Documents/Github/gbm_stock_prediction')

import gbm_helper_fxns as gbm

amd = pd.read_csv(git_path + '/data/AMD.csv')
sp500 = pd.read_csv(git_path + '/data/SPY.csv')
btc = pd.read_csv(git_path + '/data/BTC.csv')

sp500['Date']  = pd.to_datetime(sp500['Date'])
amd['Date']  = pd.to_datetime(amd['Date'])
btc['Date']  = pd.to_datetime(btc['Date'])
btc.dropna(subset = ['Adj Close'], inplace=True)
btc.reset_index(drop=True, inplace=True)

#%%
#checking normality assumption
df_asset = amd
n_train = 100
df_train = df_asset.iloc[:n_train]
df_returns = gbm.calc_returns(df_train)

mu = np.mean(df_returns)
sigma = np.std(df_returns)

plt.hist(df_returns, bins=12, density=True, alpha=0.6)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma))
plt.title("Returns Distribution")

sm.qqplot(df_returns)
plt.title("Q-Q Plot")

#%%
n = 30
dt = 1
sim = 10000
test_start = 150
size_start = 30
size_end = 110

forecast_acc = gbm.eval_n_size_forecast_acc(df_asset, dt, size_start, size_end, n, sim, test_start)
df_forecast = forecast_acc['df']
mtx_signif = gbm.paired_ttest(forecast_acc['size_acc'])
df_direction = gbm.eval_n_size_direction_acc(df_asset, dt, size_start, size_end, sim, test_start)
all_accuracy = pd.merge(df_forecast, df_direction, on = 'training_size')

#%%
# 1 sample 30 day path
n_train = 50

sim_results = gbm.multiple_one_day_GBM(df_asset, dt, n_train, n, 1, test_start)

df_sim = df_asset[['Date','Adj Close']].iloc[test_start-1:test_start+n-1]
df_sim['GBM Sim'] = sim_results

df_sim.plot(x='Date')
plt.title("Normal 30 Business Day Forecast (training set = 50)")
plt.ylabel("Price")
plt.xticks(rotation = 45)

#%%
# stationary prolonged
n = 120
test_start = n

forecast_acc = gbm.eval_n_size_forecast_acc(df_asset, dt, size_start, size_end, n, sim, test_start)
df_forecast = forecast_acc['df']
mtx_signif = gbm.paired_ttest(forecast_acc['size_acc'])
df_direction = gbm.eval_n_size_direction_acc(df_asset, dt, size_start, size_end, sim, test_start)
all_accuracy = pd.merge(df_forecast, df_direction, on = 'training_size')

#%%
#non stationary 
n = 120
test_start = n

forecast_acc = gbm.eval_n_size_forecast_acc(df_asset, dt, size_start, size_end, n, sim, test_start, method = 'moving')
df_forecast = forecast_acc['df']
mtx_signif = gbm.paired_ttest(forecast_acc['size_acc'])
df_direction = gbm.eval_n_size_direction_acc(df_asset, dt, size_start, size_end, sim, test_start, method = 'moving')
all_accuracy = pd.merge(df_forecast, df_direction, on = 'training_size')