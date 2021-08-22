# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 04:03:59 2021

@author: Ivan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from scipy.stats import norm, gaussian_kde
import statsmodels.api as sm
from scipy.stats import ttest_ind

os.chdir('D:/Documents/Github/gbm_stock_prediction')
#os.chdir('/Users/isheng/Documents/Github/gbm_stock_prediction')
amd = pd.read_csv('AMD.csv')
sp500 = pd.read_csv('SPY.csv')
btc = pd.read_csv('BTC.csv')

sp500['Date']  = pd.to_datetime(sp500['Date'])
amd['Date']  = pd.to_datetime(amd['Date'])
btc['Date']  = pd.to_datetime(btc['Date'])
btc.dropna(subset = ['Adj Close'], inplace=True)
btc.reset_index(drop=True, inplace=True)

###############################
amd = btc

n_train = 100
amd_train = amd.iloc[:n_train]

#temp
amd_returns = calc_returns(amd_train)

mu = np.mean(amd_returns)
sigma = np.std(amd_returns)

plt.hist(amd_returns, bins=12, density=True, alpha=0.6)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma))
plt.title("Returns Distribution")

sm.qqplot(amd_returns)
plt.title("Q-Q Plot")

###########################

n = 30
dt = 1
sim = 10000
test_start = n
list_acc = []
list_rmse = []
list_nrmse = []
list_mape = []
training_size = []

st = np.array(amd['Adj Close'][test_start-1:test_start-1+n].reset_index(drop=True))

for i in range(30,110,10):
    sim_results = multiple_one_day_GBM(amd, dt, i, n, sim, test_start)
    acc = forecasting_acc(st, sim_results)
    list_acc.append(acc)
    list_rmse.append(np.mean(acc['rmse']))
    list_nrmse.append(np.mean(acc['nrmse']))
    list_mape.append(np.mean(acc['mape']))
    training_size.append(i)

dim = len(training_size)
mtx_signif = np.zeros((dim,dim))
for i in range(len(list_acc)):
    group1 = list_acc[i]['mse']
    for j in reversed(range(i+1,len(list_acc))):
        group2 = list_acc[j]['mse']
        mtx_signif[j][i] = ttest_ind(group1,group2)[1]
mtx_signif = mtx_signif + mtx_signif.T - np.diag(np.diag(mtx_signif))
mtx_signif = pd.DataFrame(mtx_signif, columns = list(range(30,110,10)))
mtx_signif['index'] = list(range(30,110,10))
mtx_signif = mtx_signif.set_index('index')

n = 1
training_size = []
p_direction = []

st = np.array(amd['Adj Close'][test_start:test_start+n].reset_index(drop=True))
s0 = np.array(amd['Adj Close'][test_start-1:test_start-1+n].reset_index(drop=True))
direction = (st-s0) > 0

for i in range(30,110,10):
    sim_direction = ((multiple_one_day_GBM(amd, dt, i, n, sim, test_start).T - s0) > 0) == direction
    p_direction.append(len(sim_direction[sim_direction==True])/sim)
    training_size.append(i)

all_accuracy = pd.DataFrame(list(zip(training_size, list_rmse, list_nrmse,list_mape, p_direction)), 
                                 columns = ['training_size','Expected RMSE','Expected NRMSE','Expected MAPE','P(Correct Direction)'])

######################################
# 1 sample 30 day path
n = 30
dt = 1
sim = 1
n_train = 50
sim_results = moving_GBM(amd, dt, n_train, n, sim, test_start)

amd_sim = amd[['Date','Adj Close']].iloc[test_start-1:test_start+n-1]
amd_sim['GBM Sim'] = sim_results

amd_sim.plot(x='Date')
plt.title("Normal 30 Business Day Forecast (training set = 50)")
plt.ylabel("Price")
plt.xticks(rotation = 45)

######################
# moving test data
n = 30
dt = 1
sim = 1000
test_start = 100
list_acc = []
list_rmse = []
list_nrmse = []
list_mape = []
training_size = []

st = np.array(amd['Adj Close'][test_start:test_start+n].reset_index(drop=True))

for j in range(0,31):
    for i in range(30,110,10):
        sim_results = multiple_one_day_GBM(amd, dt, i, n, sim, test_start+j)

###########################
# stationary prolonged
n = 120
dt = 1
sim = 10000
test_start = n
list_acc = []
list_rmse = []
list_nrmse = []
list_mape = []
training_size = []

st = np.array(amd['Adj Close'][test_start-1:test_start-1+n].reset_index(drop=True))

for i in range(30,110,10):
    sim_results = multiple_one_day_GBM(amd, dt, i, n, sim, test_start)
    acc = forecasting_acc(st, sim_results)
    list_acc.append(acc)
    list_rmse.append(np.mean(acc['rmse']))
    list_nrmse.append(np.mean(acc['nrmse']))
    list_mape.append(np.mean(acc['mape']))
    training_size.append(i)

dim = len(training_size)
mtx_signif = np.zeros((dim,dim))
for i in range(len(list_acc)):
    group1 = list_acc[i]['mse']
    for j in reversed(range(i+1,len(list_acc))):
        group2 = list_acc[j]['mse']
        mtx_signif[j][i] = ttest_ind(group1,group2)[1]
mtx_signif = mtx_signif + mtx_signif.T - np.diag(np.diag(mtx_signif))
mtx_signif = pd.DataFrame(mtx_signif, columns = list(range(30,110,10)))
mtx_signif['index'] = list(range(30,110,10))
mtx_signif = mtx_signif.set_index('index')

n = 1
training_size = []
p_direction = []

st = np.array(amd['Adj Close'][test_start:test_start+n].reset_index(drop=True))
s0 = np.array(amd['Adj Close'][test_start-1:test_start-1+n].reset_index(drop=True))
direction = (st-s0) > 0

for i in range(30,110,10):
    sim_direction = ((multiple_one_day_GBM(amd, dt, i, n, sim, test_start).T - s0) > 0) == direction
    p_direction.append(len(sim_direction[sim_direction==True])/sim)
    training_size.append(i)

all_accuracy = pd.DataFrame(list(zip(training_size, list_rmse, list_nrmse,list_mape, p_direction)), 
                                 columns = ['training_size','Expected RMSE','Expected NRMSE','Expected MAPE','P(Correct Direction)'])

###########################
#non stationary 
n = 120
dt = 1
sim = 10000
list_acc = []
list_rmse = []
list_nrmse = []
list_mape = []
training_size = []

for i in range(30,110,10):
    test_start = i+2
    st = np.array(amd['Adj Close'][test_start-1:test_start-1+n].reset_index(drop=True))
    sim_results = moving_GBM(amd, dt, i, n, sim, test_start)
    acc = forecasting_acc(st, sim_results)
    list_acc.append(acc)
    list_rmse.append(np.mean(acc['rmse']))
    list_nrmse.append(np.mean(acc['nrmse']))
    list_mape.append(np.mean(acc['mape']))
    training_size.append(i)

dim = len(training_size)
mtx_signif = np.zeros((dim,dim))
for i in range(len(list_acc)):
    group1 = list_acc[i]['mse']
    for j in reversed(range(i+1,len(list_acc))):
        group2 = list_acc[j]['mse']
        mtx_signif[j][i] = ttest_ind(group1,group2)[1]
mtx_signif = mtx_signif + mtx_signif.T - np.diag(np.diag(mtx_signif))
mtx_signif = pd.DataFrame(mtx_signif, columns = list(range(30,110,10)))
mtx_signif['index'] = list(range(30,110,10))
mtx_signif = mtx_signif.set_index('index')

n = 1
training_size = []
p_direction = []

for i in range(30,110,10):
    test_start = i+2
    st = np.array(amd['Adj Close'][test_start:test_start+n].reset_index(drop=True))
    s0 = np.array(amd['Adj Close'][test_start-1:test_start-1+n].reset_index(drop=True))
    direction = (st-s0) > 0
    sim_direction = ((moving_GBM(amd, dt, i, n, sim, test_start).T - s0) > 0) == direction
    p_direction.append(len(sim_direction[sim_direction==True])/sim)
    training_size.append(i)

all_accuracy = pd.DataFrame(list(zip(training_size, list_rmse, list_nrmse,list_mape, p_direction)), 
                                 columns = ['training_size','Expected RMSE','Expected NRMSE','Expected MAPE','P(Correct Direction)'])
