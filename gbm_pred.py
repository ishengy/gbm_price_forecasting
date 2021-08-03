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

os.chdir('/Users/isheng/Downloads')
amd = pd.read_csv('/Users/isheng/Downloads/AMD.csv')
amd['Date']  = pd.to_datetime(amd['Date'])

def calc_returns(df):
    curr = df['Adj Close']
    prev = df['Adj Close'].shift(1)
    delta = (curr - prev) / prev
    return(delta)

amd_returns = calc_returns(amd)

mu = np.mean(amd_returns)
sigma = np.std(amd_returns)
n = 7
dt = 1
sim = 500
s0 = amd['Adj Close'][0]

#mu = 1
#sigma = 0.8
#n = 50
#dt = 0.1
#sim = 5
#s0 = 100

def generate_GBM(mu, sigma, dt, n, sim, s0):
    noise = np.random.normal(0, np.sqrt(dt), size=(n,sim))
    s = np.exp((mu - sigma ** 2 / 2) * dt + sigma * noise)
    s = np.vstack([np.ones(sim), s])
    s = s0 * s.cumprod(axis=0)
    return(s)

rest = generate_GBM(mu, sigma, dt, n, sim, s0)

plt.plot(rest)
plt.xlabel("$t$")
plt.ylabel("$x$")
plt.title(
    "Realizations of Geometric Brownian Motion with different variances\n $\mu=1$"
)
plt.show()
