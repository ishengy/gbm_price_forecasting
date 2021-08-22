# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 03:59:19 2021

@author: Ivan
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

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
    train_start = test_start-n_train-2
    train_end = test_start-2

    df_train = df.iloc[train_start:train_end]
    df_returns = calc_returns(df_train)

    mu = np.mean(df_returns)
    sigma = np.std(df_returns)
    
    noise = np.random.normal(0, np.sqrt(dt), size=(n,sim))
    s = np.exp((mu - sigma ** 2 / 2) * dt + sigma * noise)
    sim_results = np.multiply(np.array(df['Adj Close'][test_start-1:test_start-1+n]),s.T).T
    print(df['Date'][test_start-1:test_start-1+n])
    return(sim_results)

def moving_GBM(df, dt, n_train, n, sim, start_index):
    sim_results = np.zeros(shape=(0,sim))
    for i in range(0,n):
        test_start = start_index + i
        sim_run = multiple_one_day_GBM(df, dt, n_train, 1, sim, test_start)
        sim_results = np.append(sim_results,sim_run, axis = 0)
    print(df['Date'][test_start-1:test_start])
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