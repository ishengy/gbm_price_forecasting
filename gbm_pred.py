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

amd_returns = calc_returns(amd)

#check if returns follows a normal distribution
plot_hist(amd_returns)

mu = np.mean(amd_returns)
sigma = np.std(amd_returns)
n = 7
dt = 1
sim = 1000
s0 = amd['Adj Close'][0]

def generate_GBM(mu, sigma, dt, n, sim, s0):
    noise = np.random.normal(0, np.sqrt(dt), size=(n,sim))
    s = np.exp((mu - sigma ** 2 / 2) * dt + sigma * noise)
    s = np.vstack([np.ones(sim), s])
    s = s0 * s.cumprod(axis=0)
    return(s)

test = generate_GBM(mu, sigma, dt, n, sim, s0)
plot_hist(test[1])

#plt.plot(test)
#plt.xlabel("$t$")
#plt.ylabel("$x$")
#plt.show()

dayMean = test.mean(axis=1)
dayVar = test.std(axis=1)

#randomly draw from x=day, N(dayMean[x],dayVar[x]) to generate line plot?