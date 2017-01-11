# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:52:01 2017

@author: ZFang
"""

import pandas as pd
import os 
import numpy as np
from datetime import datetime
import scipy as sp
import scipy.optimize as scopt
import scipy.stats as spstats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def read_data():
    # os.getcwd()
    os.chdir('C:\\Users\\ZFang\\Desktop\\TeamCo\\Portfolio-Optimization')
    fund_df = pd.read_excel('fund_data.xlsx',index_col='Date')
    return fund_df
    
# calculate annual returns
def calc_annual_returns(daily_returns):
    grouped = np.exp(daily_returns.groupby(
        lambda date: date.year).sum())-1
    return grouped   
    
    
    
def calc_portfolio_var(returns, weights=None):
    if weights is None:
        weights = np.ones(returns.columns.size)/ returns.columns.size
    sigma = np.cov(returns.T, ddof=0)
    var = (weights * sigma * weights.T).sum()
    return var
    
def sharpe_ratio(returns, weights = None, risk_free_rate=0.015):
    n = returns.columns.size
    if weights is None: weights = np.ones(n)/n
    print('The weight is ' + str(weights))
    # get the portfolio variance
    var = calc_portfolio_var(returns, weights)
    # and the means of the stocks in the portfolio
    means = returns.mean()
    # and returns the sharpe ratio
    print('The Sharpe Ratio is ' + str((means.dot(weights)-risk_free_rate)/np.sqrt(var)))
    return (means.dot(weights)-risk_free_rate)/np.sqrt(var)    
    
def negative_sharpe_ratio(weights, 
                          returns, 
                          risk_free_rate):
    """
    Given n-1 weights, return a negative sharpe ratio
    """
    return -sharpe_ratio(returns, weights, risk_free_rate)   
    
    
def optimize_portfolio(returns, risk_free_rate):
    """ 
    Performs the optimization
    """
    # start with equal weights
    w0 = np.ones(returns.columns.size, 
                 dtype=float) * 1.0 / returns.columns.size
    # minimize the negative sharpe value
    constraints = ({'type': 'eq', 
                'fun': lambda w0: w0[0]+w0[1]-0.2})
    bounds=((0,1),(0,1),(0,1),(0,1),(0,1))
    w1 = scopt.minimize(negative_sharpe_ratio, 
                    w0, args=(returns, risk_free_rate),
                    method='SLSQP', constraints = constraints,
                    bounds = bounds).x
    print('Reach to the last step')
    print('The final w1 is ' + str(w1))
    # and calculate the final, optimized, sharpe ratio
    final_sharpe = sharpe_ratio(returns, w1, risk_free_rate)
    return (w1, final_sharpe)    
    
def objfun(W,R,target_ret):
    stock_mean = np.mean(R,axis=0)
    port_mean = np.dot(W,stock_mean) # portfolio mean
    cov = np.cov(R.T) # var-cov matrix
    port_var = np.dot(np.dot(W,cov),W.T) # portfolio variance
    penalty = 2000* abs(port_mean-target_ret) # penalty 4 deviation
    return np.sqrt(port_var) + penalty # objective function
    

def calc_efficient_frontier(returns):
    result_means = []
    result_stds = []
    result_weights = []
    
    means = returns.mean()
    min_mean, max_mean = means.min(), means.max()
    
    nstocks = returns.columns.size
    
    for r in np.linspace(min_mean, max_mean, 100):
        weights = np.ones(nstocks)/nstocks
        bounds = [(0,1) for i in np.arange(nstocks)]
        constraints = ({'type': 'eq', 
                        'fun': lambda W: np.sum(W) - 1})
        results = scopt.minimize(objfun, weights, (returns, r), 
                                 method='SLSQP', 
                                 constraints = constraints,
                                 bounds = bounds)
        if not results.success: # handle error
            raise Exception(results.message)
        result_means.append(np.round(r,4)) # 4 decimal places
        std_=np.round(np.std(np.sum(returns*results.x,axis=1)),6)
        result_stds.append(std_)
        
        result_weights.append(np.round(results.x, 5))
    return {'Means': result_means, 
            'Stds': result_stds, 
            'Weights': result_weights}
            
def plot_efficient_frontier(frontier_data):
    plt.figure(figsize=(12,8))
    plt.title('Efficient Frontier')
    plt.xlabel('Standard Deviation of the porfolio (Risk))')
    plt.ylabel('Return of the portfolio')
    plt.plot(frontier_data['Stds'], frontier_data['Means'], '--'); 
    plt.savefig('5104OS_09_20.png', bbox_inches='tight', dpi=300)
    
if __name__ == '__main__':    
    fund_df = read_data()
    # Calculate the Annual Return of All Funds
    annual_fund_return = calc_annual_returns(fund_df)
    # calculate our portfolio variance (equal weighted)
    calc_portfolio_var(annual_fund_return)
    # calculate equal weighted sharpe ratio
    eql_sharpe = sharpe_ratio(annual_fund_return)
    # optimize our portfolio
    opt_weight = optimize_portfolio(annual_fund_return.iloc[:,:5], 0.0003)
    
    # calc_portfolio_var(annual_fund_return.iloc[:,:5])
    '''
    n = 5
    weights = np.ones(n)/n
    # negative_sharpe_ratio(weights,annual_fund_return.iloc[:,:5],0.0003)
    returns = annual_fund_return.iloc[:,:5]
    
    
    
    w1 = scopt.minimize(negative_sharpe_ratio, 
                weights, args=(returns, 0.0002)).x
    '''