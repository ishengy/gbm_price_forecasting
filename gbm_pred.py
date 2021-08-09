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

def generate_GBM(mu, sigma, dt, n, sim, s0):
    noise = np.random.normal(0, np.sqrt(dt), size=(n,sim))
    s = np.exp((mu - sigma ** 2 / 2) * dt + sigma * noise)
    s = np.vstack([np.ones(sim), s])
    s = s0 * s.cumprod(axis=0)
    return(s)


n_train = 100
amd_train = amd.iloc[:n_train]
amd_returns = calc_returns(amd_train)

mu = np.mean(amd_returns)
sigma = np.std(amd_returns)

plt.hist(amd_returns, bins=12, density=True, alpha=0.6)
xmin, xmax = plt.xlim()
x_axis = np.linspace(xmin, xmax, 100)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma))
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
    
    plt.hist(sim_results[1], bins=12, density=True)
    xmin, xmax = plt.xlim()
    x_axis = np.linspace(xmin, xmax, 100)
    plt.plot(x_axis, norm.pdf(x_axis, sim_avg[0], sim_std[0]))
    plt.title("One-Day Simulations")
    plt.xlabel("AMD Price")
    
    actual_pdf = norm.pdf(st, loc=sim_avg, scale=sim_std)
    actual_cdf = norm.cdf(st, loc=sim_avg, scale=sim_std)
    
    list_mse.append(mse(st, sim_avg))
    list_mape.append(mape(st, sim_avg))

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
plt.xlabel("AMD Price")

actual_pdf = norm.pdf(st, loc=sim_avg, scale=sim_std)
actual_cdf = norm.cdf(st, loc=sim_avg, scale=sim_std)

mse(st, sim_avg)
mape(st, sim_avg)
##########################
#7 day simuations
n = 7
dt = 1
sim = 100
s0 = amd['Adj Close'][n_train]

sim_results = generate_GBM(mu, sigma, dt, n, sim, s0)
st = amd['Adj Close'][n_train+1:n_train+1+n].reset_index(drop=True)
sim_avg = np.mean(sim_results, axis=1)[1:n+1]
sim_std = np.std(sim_results, axis=1)[1:n+1]

amd_sim = amd[['Date','Adj Close']].iloc[n_train+1:n_train+1+n]
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

###########################
# i like this - replace for 1 day prediction? and say 1 day at a time prediction?
n = 30
dt = 1
sim = 1
st = amd['Adj Close'][n_train]

def extended_one_day_GBM(df, dt, n_train, n, sim):
    df_train = df.iloc[:n_train]
    df_returns = calc_returns(df_train)
    
    mu = np.mean(df_returns)
    sigma = np.std(df_returns)
    
    noise = np.random.normal(0, np.sqrt(dt), size=(n,sim))
    s = np.exp((mu - sigma ** 2 / 2) * dt + sigma * noise)
    sim_results = np.multiply(np.array(df['Adj Close'][n_train-1:n_train+n-1]),s.T).T
    return(sim_results)

sim_results = extended_one_day_GBM(amd, dt, n_train, n, sim)

amd_sim = amd[['Date','Adj Close']].iloc[n_train:n_train+n]
amd_sim['GBM Sim'] = sim_results
#plot_hist(test[1])

print(mse(st, sim_results))
print(mape(st, sim_results))

amd_sim.plot(x='Date')
plt.title("7 Business Day Forecast")
plt.ylabel("AMD Price")