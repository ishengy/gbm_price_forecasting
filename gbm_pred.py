#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 02:42:39 2021
@author: isheng
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

def calc_returns(df):
    curr = df['Adj Close']
    prev = df['Adj Close'].shift(1)
    delta = (curr - prev) / prev
    return(delta)

def plot_hist(data):
    s = pd.Series(data)
    s.plot.hist(bins=12, density = True)

def mse(actual, pred):
    return(np.square(np.subtract(actual,pred.T)).mean(axis=0))

def rmse(actual, pred):
    return(np.sqrt(mse(actual, pred)))

def nrmse(actual, pred):
    return(rmse(actual, pred)/np.mean(actual))

def mape(actual, pred): 
    return(np.mean(np.abs((actual - pred.T) / actual)))

def forecasting_acc(actual, pred):
    d = dict()
    mape = np.abs((actual - pred.T) / actual).mean(axis = 0)
    mse = np.square(np.subtract(actual,pred.T)).mean(axis = 0)
    rmse = np.sqrt(mse)
    nrmse = rmse/np.mean(actual)
    
    d['mape'] = mape
    d['mse'] = mse
    d['rmse'] = rmse
    d['nrmse'] = nrmse
    return(d)

def multiple_one_day_GBM(df, dt, n_train, n, sim, test_start):
    #start for loop here? and set size of noise to (1,sim)? maybe not tbh
    #for(i in range(0,days)):
    train_start = test_start-n_train-2
    train_end = test_start-2
    
    df_train = df.iloc[train_start:train_end]
    df_returns = calc_returns(df_train)

    mu = np.mean(df_returns)
    sigma = np.std(df_returns)
    
    noise = np.random.normal(0, np.sqrt(dt), size=(n,sim))
    s = np.exp((mu - sigma ** 2 / 2) * dt + sigma * noise)
    sim_results = np.multiply(np.array(df['Adj Close'][test_start-1:test_start-1+n]),s.T).T
    return(sim_results)

def kde_GBM(df, dt, n_train, n, sim, test_start):
    train_start = test_start-n_train-2
    train_end = test_start-2
    
    df_train = df.iloc[train_start:train_end]
    df_returns = calc_returns(df_train)
    
    kde = gaussian_kde(df_returns[1:])
    noise = (kde.resample(n*sim)).reshape(n,sim)
    s = np.exp(noise)
    sim_results = np.multiply(np.array(df['Adj Close'][test_start-1:test_start-1+n]),s.T).T
    return(sim_results)

#amd = sp500

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
#############################
# KDE

kde = gaussian_kde(amd_returns[1:], bw_method = 'silverman')
x_axis = np.linspace(xmin, xmax, 100)
den = kde.evaluate(x_axis)
plt.figure()
plt.hist(amd_returns, bins = 12, density = True)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma), label = 'Normal')
plt.plot(x_axis, den, label= 'KDE')
plt.legend()
plt.title('Returns Distribution')
plt.show()

test = kde.resample(1000000).T
plt.hist(test, density = True)

###########################

n = 30
dt = 1
sim = 100000
test_start = 200
list_acc = []
list_rmse = []
list_nrmse = []
list_mape = []
training_size = []

st = np.array(amd['Adj Close'][test_start-1:test_start-1+n].reset_index(drop=True))

for i in range(30,110,10):
    sim_results = kde_GBM(amd, dt, i, n, sim, test_start)
    acc = forecasting_acc(st, sim_results)
    list_acc.append(acc)
    list_rmse.append(np.mean(acc['rmse']))
    list_nrmse.append(np.mean(acc['nrmse']))
    list_mape.append(np.mean(acc['mape']))
    training_size.append(i)

mtx_signif = np.zeros((8,8))
for i in range(len(list_acc)):
    group1 = list_acc[i]['mse']
    for j in reversed(range(i+1,len(list_acc))):
        group2 = list_acc[j]['mse']
        mtx_signif[j][i] = ttest_ind(group1,group2)[1]

#plt.figure()
#plt.hist(sim_results[:,0], label = 'Sample Simulation', density = True, alpha=0.8)
#plt.hist(st, label = 'Test', density = True, alpha=0.8)
#plt.title('SPY Test vs Simulation')
#plt.legend()
#plt.show()

#train_start = test_start-n_train-2
#train_end = test_start-2

#plt.hist(amd['Adj Close'].iloc[train_start:train_end], label = 'Training', density = True, alpha=0.8)
#plt.hist(st, label = 'Test', density = True, alpha=0.8)
#plt.title('SPY Test vs Training')
#plt.legend()

n = 1
training_size = []
p_direction = []

st = np.array(amd['Adj Close'][test_start:test_start+n].reset_index(drop=True))
s0 = np.array(amd['Adj Close'][test_start-1:test_start-1+n].reset_index(drop=True))
direction = (st-s0) > 0

for i in range(30,110,10):
    sim_direction = ((kde_GBM(amd, dt, i, n, sim, test_start).T - s0) > 0) == direction
    p_direction.append(len(sim_direction[sim_direction==True])/sim)
    training_size.append(i)

all_accuracy = pd.DataFrame(list(zip(training_size, list_rmse, list_nrmse,list_mape, p_direction)), 
                                 columns = ['training_size','Expected RMSE','Expected NRMSE','Expected MAPE','P(Correct Direction)'])

######################################
# 1 sample 30 day path
n = 30
dt = 1
sim = 1
n_train = 100
sim_results = multiple_one_day_GBM(amd, dt, n_train, n, sim, test_start)

amd_sim = amd[['Date','Adj Close']].iloc[test_start:test_start+n]
amd_sim['GBM Sim'] = sim_results

amd_sim.plot(x='Date')
plt.title("Normal 30 Business Day Forecast (training set = 100)")
plt.ylabel("Price")
#plt.xticks(rotation = 45)

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

######################
# scratch pad - Ignore
train_start = 0
train_end = test_start-2
    
df_train = amd.iloc[train_start:train_end]
df_returns = calc_returns(df_train)

mu = np.mean(df_returns)
sigma = np.std(df_returns)

noise = np.random.normal(0, np.sqrt(dt), size=(100000,1))
e = (mu - sigma ** 2 / 2) * dt + sigma * noise
s = np.exp(e)

kde = gaussian_kde(df_returns[1:])
noise1 = (kde.resample(1*100000)).reshape(100000,1)
s1 = np.exp(noise1)

x_axis = np.linspace(xmin, xmax, 100)
den = kde.evaluate(x_axis)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma), label = 'Normal')
plt.plot(x_axis,den)
plt.hist(df_returns, density=True, bins=15, alpha=0.6)
plt.hist(noise1, density=True, bins=15, alpha=0.6)

##############

n_train = 30
df=amd
train_start = test_start-n_train-1
train_end = test_start-1
    
df_train = df.iloc[train_start:train_end]
df_returns = calc_returns(df_train)
    
print(df_returns)
print(df['Date'][test_start-1:test_start-1+n])
st = amd[['Date','Adj Close']][test_start-1:test_start-1+n]
print(st)

mu = np.mean(df_returns)
sigma = np.std(df_returns)
    
noise = np.random.normal(0, np.sqrt(dt), size=(n,sim))
s = np.exp((mu - sigma ** 2 / 2) * dt + sigma * noise)
sim_results = np.multiply(np.array(df['Adj Close'][test_start-1:test_start-1+n]),s.T).T