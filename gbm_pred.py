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

#os.chdir('D:/Documents/Github/gbm_stock_prediction')
#amd = pd.read_csv('AMD.csv')

os.chdir('/Users/isheng/Downloads')
amd = pd.read_csv('/Users/isheng/Downloads/AMD.csv')

amd['Date']  = pd.to_datetime(amd['Date'])
amd_train = amd.iloc[:100]

def calc_returns(df):
    curr = df['Adj Close']
    prev = df['Adj Close'].shift(1)
    delta = (curr - prev) / prev
    return(delta)

def plot_hist(data):
    s = pd.Series(data)
    s.plot.hist(bins=12)

amd_returns = calc_returns(amd_train)

#check if returns follows a normal distribution
#bootstrap if distribution looks funky?
#plot normal distribution line over it for comparison?
plot_hist(amd_returns)

mu = np.mean(amd_returns)
sigma = np.std(amd_returns)
n = 7
dt = 1
sim = 1000
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
#amd_test = amd.iloc[100:107]
sim_avg = np.mean(sim_results, axis=1)[1:n+1]
amd_sim = amd[['Date','Adj Close']].iloc[101:n+101]
amd_sim['GBM'] = sim_avg
#plot_hist(test[1])

amd_sim.plot(x='Date')
# predict 7 days with x number of training data
# average it across and plot real vs average???
# compare MSE and the probability of an up vs down prediction using Montes day by day?

dayMean = test.mean(axis=1)
dayVar = test.std(axis=1)

#randomly draw from x=day, N(dayMean[x],dayVar[x]) to generate line plot?
plt.plot(dayMean)
plt.plot(amd['Adj Close'])
plt.show()

amd.plot(x='Date',y='Adj Close')
amd_sim.plot(x='Date',y='Adj Close')
