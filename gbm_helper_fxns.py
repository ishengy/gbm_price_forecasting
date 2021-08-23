# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 03:59:19 2021

@author: Ivan
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import ttest_ind

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
    return(sim_results)

def moving_GBM(df, dt, n_train, n, sim, start_index):
    sim_results = np.zeros(shape=(0,sim))
    for i in range(0,n):
        test_start = start_index + i
        sim_run = multiple_one_day_GBM(df, dt, n_train, 1, sim, test_start)
        sim_results = np.append(sim_results,sim_run, axis = 0)
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

def paired_ttest(list_acc):
    dim = len(list_acc)
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
    return(mtx_signif)

def eval_n_size_forecast_acc(df, dt, size_start, size_end, n, sim, test_start, method = 'stationary'):
    d = dict()
    list_acc = []
    list_rmse = []
    list_nrmse = []
    list_mape = []
    training_size = []
    
    if method == 'stationary':
        st = np.array(df['Adj Close'][test_start-1:test_start-1+n].reset_index(drop=True))
        
    for i in range(size_start,size_end,10):
        if method != 'stationary':
            test_start = i+2
            st = np.array(df['Adj Close'][test_start-1:test_start-1+n].reset_index(drop=True))
            sim_results = moving_GBM(df, dt, i, n, sim, test_start)
        else:
            sim_results = multiple_one_day_GBM(df, dt, i, n, sim, test_start)
        
        acc = forecasting_acc(st, sim_results)
        list_acc.append(acc)
        list_rmse.append(np.mean(acc['rmse']))
        list_nrmse.append(np.mean(acc['nrmse']))
        list_mape.append(np.mean(acc['mape']))
        training_size.append(i)
    
    forecast_acc = pd.DataFrame(list(zip(training_size, list_rmse, list_nrmse,list_mape)), 
                                 columns = ['training_size','Expected RMSE','Expected NRMSE','Expected MAPE'])

    d['training_size'] = training_size
    d['size_acc'] = list_acc
    d['size_mape'] = list_mape
    d['size_rmse'] = list_rmse
    d['size_nrmse'] = list_nrmse
    d['df'] = forecast_acc
    return(d)

def eval_n_size_direction_acc(df, dt, size_start, size_end, sim, test_start, method = 'stationary'):
    training_size = []
    p_direction = []
    
    if method == 'stationary':
        st = np.array(df['Adj Close'][test_start:test_start+1].reset_index(drop=True))
        s0 = np.array(df['Adj Close'][test_start-1:test_start].reset_index(drop=True))
        direction = (st-s0) > 0
    
    for i in range(size_start,size_end,10):
        if method != 'stationary':
            test_start = i+2
            st = np.array(df['Adj Close'][test_start:test_start+1].reset_index(drop=True))
            s0 = np.array(df['Adj Close'][test_start-1:test_start].reset_index(drop=True))
            direction = (st-s0) > 0
            sim_direction = ((moving_GBM(df, dt, i, 1, sim, test_start).T - s0) > 0) == direction
        else:
            sim_direction = ((multiple_one_day_GBM(df, dt, i, 1, sim, test_start).T - s0) > 0) == direction
        p_direction.append(len(sim_direction[sim_direction==True])/sim)
        training_size.append(i)
    
    direction_acc = pd.DataFrame(list(zip(training_size, p_direction)), 
                                     columns = ['training_size','P(Correct Direction)'])
    return(direction_acc)
