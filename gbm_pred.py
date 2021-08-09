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
from scipy.stats import norm

os.chdir('D:/Documents/Github/gbm_stock_prediction')
amd = pd.read_csv('AMD.csv')

#os.chdir('/Users/isheng/Downloads')
#amd = pd.read_csv('/Users/isheng/Downloads/AMD.csv')

amd['Date']  = pd.to_datetime(amd['Date'])

def calc_returns(df):
    curr = df['Adj Close']
    prev = df['Adj Close'].shift(1)
    delta = (curr - prev) / prev
    return(delta)

def plot_hist(data):
    s = pd.Series(data)
    s.plot.hist(bins=12)

def mse(actual, pred):
    return(np.square(np.subtract(actual,pred).mean()))

def mape(actual, pred): 
    return np.mean(np.abs((actual - pred) / actual)) 

amd_train = amd.iloc[:100]
amd_returns = calc_returns(amd_train)

#check if returns follows a normal distribution
#bootstrap if distribution looks funky?
#plot normal distribution line over it for comparison?

mu = np.mean(amd_returns)
sigma = np.std(amd_returns)

plt.hist(amd_returns, bins=12, density=True, alpha=0.6)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma))

###########################
n = 1
dt = 1
sim = 10000
s0 = amd['Adj Close'][100]
    
#with single day forcasting, replace s0 with the real s(t-1)????
#with multi day forecasting, keep as is?
def generate_GBM(mu, sigma, dt, n, sim, s0):
    noise = np.random.normal(0, np.sqrt(dt), size=(n,sim))
    s = np.exp((mu - sigma ** 2 / 2) * dt + sigma * noise)
    s = np.vstack([np.ones(sim), s])
    s = s0 * s.cumprod(axis=0)
    return(s)

sim_results = generate_GBM(mu, sigma, dt, n, sim, s0)
st = amd['Adj Close'][101]
sim_avg = np.mean(sim_results, axis=1)[1:n+1]
sim_std = np.std(sim_results, axis=1)[1:n+1]

plt.hist(sim_results[1], bins=12, density=True)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
plt.plot(x_axis, norm.pdf(x_axis, sim_avg, sim_std))
plt.title("One-Day Simulations")
plt.xlabel("AMD Price")

actual_pdf = norm.pdf(st, loc=sim_avg, scale=sim_std)
actual_cdf = norm.cdf(st, loc=sim_avg, scale=sim_std)

mse(st, sim_avg)
mape(st, sim_avg)
##########################

n = 7
dt = 1
sim = 10000
s0 = amd['Adj Close'][100]

sim_results = generate_GBM(mu, sigma, dt, n, sim, s0)
st = amd['Adj Close'][101:108].reset_index(drop=True)
sim_avg = np.mean(sim_results, axis=1)[1:n+1]
sim_std = np.std(sim_results, axis=1)[1:n+1]

amd_sim = amd[['Date','Adj Close']].iloc[101:n+101]
amd_sim['GBM Average'] = sim_avg
#plot_hist(test[1])

amd_sim.plot(x='Date')
plt.title("7 Business Day Forecast")
plt.ylabel("AMD Price")

mse(st, sim_avg)
mape(st, sim_avg)

actual_pdf = norm.pdf(st, loc=sim_avg, scale=sim_std)
actual_cdf = norm.cdf(st, loc=sim_avg, scale=sim_std)

plt.hist(sim_results[7], bins=12, density=True)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
plt.plot(x_axis, norm.pdf(x_axis, sim_avg[6], sim_std[6]))
plt.title("One-Week Simulations")

# predict 7 days with x number of training data
# average it across and plot real vs average???
# compare MSE and the probability of an up vs down prediction using Montes day by day?
