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

os.chdir('D:/Documents/Github/gbm_stock_prediction')
#os.chdir('/Users/isheng/Documents/Github/gbm_stock_prediction')
amd = pd.read_csv('AMD.csv')
sp500 = pd.read_csv('SPY.csv')
btc = pd.read_csv('BTC.csv')

sp500['Date']  = pd.to_datetime(sp500['Date'])
amd['Date']  = pd.to_datetime(amd['Date'])
btc['Date']  = pd.to_datetime(btc['Date'])
btc.dropna(subset = ['Adj Close'], inplace=True)

def calc_returns(df):
    curr = df['Adj Close']
    prev = df['Adj Close'].shift(1)
    delta = (curr - prev) / prev
    return(delta)

def plot_hist(data):
    s = pd.Series(data)
    s.plot.hist(bins=12, density = True)

# Delete? Replaced by nrmse()
def mse(actual, pred):
    return(np.square(np.subtract(actual,pred.T).mean(axis=0)))

def nrmse(actual, pred):
    mse = np.square(np.subtract(actual,pred.T))
    rmse = np.sqrt(mse)
    nrmse = rmse/np.mean(actual)
    return(nrmse.mean(axis=0))

def mape(actual, pred): 
    return np.mean(np.abs((actual - pred.T) / actual)) 

# Delete? Replaced by multiple_one_day_GBM()
def generate_GBM(mu, sigma, dt, n, sim, s0):
    noise = np.random.normal(0, np.sqrt(dt), size=(n,sim))
    s = np.exp((mu - sigma ** 2 / 2) * dt + sigma * noise)
    s = np.vstack([np.ones(sim), s])
    s = s0 * s.cumprod(axis=0)
    return(s)

amd = btc

n_train = 100
amd_train = amd.iloc[:n_train]

#temp
amd_returns = calc_returns(amd)

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

plt.figure()
kde = gaussian_kde(amd_returns[1:])
x_axis = np.linspace(xmin, xmax, 100)
den = kde.evaluate(x_axis)
plt.figure()
plt.hist(amd_returns, bins = 12, density = True)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma), label = 'Normal')
plt.plot(x_axis, den, label= 'KDE')
plt.legend()
plt.title('Returns Distribution')
plt.show()

test = kde.resample(10000).T
plt.hist(test, density = True)

# Updated - Retest functionality
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

###########################
#Delete?
n = 1
dt = 1
sim = 10000
s0 = amd['Adj Close'][n_train]

sim_results = generate_GBM(mu, sigma, dt, n, sim, s0)
st = amd['Adj Close'][n_train+1]
sim_avg = np.mean(sim_results, axis=1)[1:n+1]
sim_std = np.std(sim_results, axis=1)[1:n+1]

plt.hist(sim_results[1], bins=12, density=True)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
plt.plot(x_axis, norm.pdf(x_axis, sim_avg, sim_std))
plt.title("One-Day Simulations")
plt.xlabel("Price")

actual_pdf = norm.pdf(st, loc=sim_avg, scale=sim_std)
actual_cdf = norm.cdf(st, loc=sim_avg, scale=sim_std)

###########################
def multiple_one_day_GBM(df, dt, n_train, n, sim, test_start):
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

n = 30
dt = 1
sim = 100000
test_start = 200
list_nrmse = []
list_mape = []
training_size = []

st = np.array(amd['Adj Close'][test_start:test_start+n].reset_index(drop=True))

for i in range(30,110,10):
    sim_results = kde_GBM(amd, dt, i, n, sim, test_start)
    list_nrmse.append(np.mean(nrmse(st, sim_results).T))
    list_mape.append(mape(st, sim_results).T)
    training_size.append(i)

plt.figure()
plt.hist(sim_results[:,0], label = 'Sample Simulation', density = True, alpha=0.8)
plt.hist(st, label = 'Test', density = True, alpha=0.8)
plt.title('SPY Test vs Simulation')
plt.legend()
plt.show()

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

all_accuracy = pd.DataFrame(list(zip(training_size,list_nrmse,list_mape, p_direction)), 
                                 columns = ['training_size','Expected NRMSE','Expected MAPE','P(Correct Direction)'])

######################################
n = 30
dt = 1
sim = 1
n_train = 60
sim_results = kde_GBM(amd, dt, n_train, n, sim, test_start)

amd_sim = amd[['Date','Adj Close']].iloc[test_start:test_start+n]
amd_sim['GBM Sim'] = sim_results

amd_sim.plot(x='Date')
plt.title("KDE 30 Business Day Forecast (training set = 30)")
plt.ylabel("Price")

######################

amd_train = amd.iloc[:n_train]
amd_returns = calc_returns(amd_train)

mu = np.mean(amd_returns)
sigma = np.std(amd_returns)

plt.hist(amd_returns, bins=12, density=True, alpha=0.6)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma))
plt.xlabel("Returns")
plt.title("Return Distribution")