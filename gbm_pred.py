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
from sklearn.neighbors import KernelDensity

os.chdir('D:/Documents/Github/gbm_stock_prediction')
amd = pd.read_csv('AMD.csv')
sp500 = pd.read_csv('SPY.csv')
btc = pd.read_csv('BTC.csv')
#os.chdir('/Users/isheng/Downloads')
#amd = pd.read_csv('/Users/isheng/Downloads/AMD.csv')
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
    s.plot.hist(bins=12)

def mse(actual, pred):
    return(np.square(np.subtract(actual,pred).mean(axis=0)))

def mape(actual, pred): 
    return np.mean(np.abs((actual - pred) / actual)) 

def generate_GBM(mu, sigma, dt, n, sim, s0):
    noise = np.random.normal(0, np.sqrt(dt), size=(n,sim))
    s = np.exp((mu - sigma ** 2 / 2) * dt + sigma * noise)
    s = np.vstack([np.ones(sim), s])
    s = s0 * s.cumprod(axis=0)
    return(s)

#temp
amd = amd
####

n_train = 100
amd_train = amd.iloc[:n_train]
amd_returns = calc_returns(btc)

mu = np.mean(amd_returns)
sigma = np.std(amd_returns)

plt.hist(amd_returns, bins=15, density=True, alpha=0.6)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma))
plt.title("Returns Distribution")

sm.qqplot(amd_returns)
plt.title("Q-Q Plot")
#############################
# for btc
kde = gaussian_kde(amd_returns[1:])
x_axis = np.linspace(xmin, xmax, 100)
den = kde.evaluate(x_axis)
plt.figure()
plt.hist(amd_returns, bins = 15, density = True)
plt.plot(x_axis, den)
plt.show()

test = kde.resample(10000).T
plt.hist(test)

###########################

n = 7
dt = 1
sim = 10000
list_mse = []
list_mape = []
for i in range(30,210,10):
    n_train = i
    amd_train = amd.iloc[:n_train]
    amd_returns = calc_returns(amd_train)
    
    mu = np.mean(amd_returns)
    sigma = np.std(amd_returns)
    s0 = amd['Adj Close'][n_train]
    
    sim_results = generate_GBM(mu, sigma, dt, n, sim, s0)
    st = amd['Adj Close'][n_train+1]
    sim_avg = np.mean(sim_results, axis=1)[1:n+1]
    sim_std = np.std(sim_results, axis=1)[1:n+1]
    
    actual_pdf = norm.pdf(st, loc=sim_avg, scale=sim_std)
    actual_cdf = norm.cdf(st, loc=sim_avg, scale=sim_std)
    
    list_mse.append(mse(st, sim_avg))
    list_mape.append(mape(st, sim_avg))

plt.hist(sim_results[1], bins=12, density=True)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
plt.plot(x_axis, norm.pdf(x_axis, sim_avg[0], sim_std[0]))
plt.title("One-Day Simulations")
plt.xlabel("AMD Price")
###########################

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

mse(st, sim_avg)
mape(st, sim_avg)

###########################
n = 7
dt = 1
sim = 10000

def extended_one_day_GBM(df, dt, n_train, n, sim):
    df_train = df.iloc[:n_train]
    df_returns = calc_returns(df_train)
    
    mu = np.mean(df_returns)
    sigma = np.std(df_returns)
    
    noise = np.random.normal(0, np.sqrt(dt), size=(n,sim))
    s = np.exp((mu - sigma ** 2 / 2) * dt + sigma * noise)
    sim_results = np.multiply(np.array(df['Adj Close'][n_train-1:n_train+n-1]),s.T).T
    return(sim_results)

list_mse = []
list_mape = []
training_size = []
for i in range(30,110,10):
    st = np.array(amd['Adj Close'][i:i+n].reset_index(drop=True))
    sim_results = extended_one_day_GBM(amd, dt, i, n, sim)
    list_mse.append(np.mean(mse(st, sim_results.T).T))
    list_mape.append(mape(st, sim_results.T).T)
    training_size.append(i)
forecast_accuracy = pd.DataFrame(list(zip(training_size,list_mse,list_mape)), 
                                 columns = ['training_size','Expected MSE','Expected MAPE'])

######################################
n = 1
dt = 1
sim = 10000

training_size = []
p_direction = []
for i in range(30,110,10):
    st = np.array(amd['Adj Close'][i:i+n].reset_index(drop=True))
    s0 = np.array(amd['Adj Close'][i-1:i-1+n].reset_index(drop=True))
    direction = (st-s0) > 0
    sim_direction = ((extended_one_day_GBM(amd, dt, i, n, sim).T - s0) > 0) == direction
    p_direction.append(len(sim_direction[sim_direction==True])/sim)
    training_size.append(i)

direction_accuracy = pd.DataFrame(list(zip(training_size,p_direction)), 
                                 columns = ['training_size','P(Correct Direction)'])

######################################
n = 7
dt = 1
sim = 1
n_train = 70
sim_results = extended_one_day_GBM(amd, dt, n_train, n, sim)

amd_sim = amd[['Date','Adj Close']].iloc[n_train:n_train+n]
amd_sim['GBM Sim'] = sim_results

amd_sim.plot(x='Date')
plt.title("7 Business Day Forecast (training set = 70)")
plt.ylabel("AMD Price")

amd_train = amd.iloc[:n_train]
amd_returns = calc_returns(amd_train)

mu = np.mean(amd_returns)
sigma = np.std(amd_returns)

plt.hist(amd_returns, bins=12, density=True, alpha=0.6)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma))
plt.xlabel("AMD Returns")
plt.title("AMD Return Distribution")