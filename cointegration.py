#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:37:27 2019

@author: ioannismilas
"""
# https://medium.com/auquan/pairs-trading-data-science-7dbedafcfe5a

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

np.random.seed(107)
import matplotlib.pyplot as plt

Xreturns = np.random.normal(0, 1, 100)

X = pd.Series(np.cumsum(Xreturns), name = 'X') + 50
X.plot(figsize = (15, 7))
plt.show()

noise = np.random.normal(0,1,100)
Y = X + 5 + noise
Y.name = 'Y'

pd.concat([X, Y], axis = 1).plot(figsize = (15,7))
plt.show()

(Y/X).plot(figsize=(15,7))
plt.axhline((Y/X).mean(), color = 'red', linestyle = '--')
plt.xlabel('Time')
plt.legend(['Price Ratio', 'Mean'])
plt.show()

score, pvalue, _ = coint(X, Y)
print(pvalue)
X.corr(Y)

# series correlated, not cointegrated

ret1 = np.random.normal(1, 1, 100)
ret2 = np.random.normal(2, 1, 100)

s1 = pd.Series(np.cumsum(ret1), name = 'X')
s2 = pd.Series(np.cumsum(ret2), name = 'Y')

pd.concat([s1, s2], axis = 1).plot(figsize = (15,7))
plt.show()

print('Correlation: ' + str(s1.corr(s2)))
score, pvalue, _ = coint(s1, s2)
print('Cointegration test p-value: ' + str(pvalue))

# cointegrated, not correlated

Y2 = pd.Series(np.random.normal(0,1,800), name = 'Y2') + 20
Y3 = Y2.copy()

Y3[0:100] = 30
Y3[100:200] = 10
Y3[200:300] = 30
Y3[300:400] = 10
Y3[400:500] = 30
Y3[500:600] = 10
Y3[600:700] = 30
Y3[700:800] = 10

Y2.plot(figsize = (15,7))
Y3.plot()
plt.ylim([0,40])
plt.show()

print('Correlation: ' + str(Y2.corr(Y3)))
score, pvalue, _ = coint(Y2, Y3)
print('Cointegration test p-value: ' + str(pvalue))

##

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n,n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.02:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs
    
import quandl
import datetime
    
def get(tickers, startdate, enddate):
    def data(ticker):
        return(quandl.get(ticker, start_date = startdate, end_date = enddate))
    datas= map(data, tickers)
    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))
    
tickers = ['WIKI/AAPL', 'WIKI/MSFT', 'WIKI/IBM', 'WIKI/GOOGL', 'WIKI/FB', 'WIKI/EBAY', 'WIKI/AMD', 'WIKI/ADBE']
all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2017, 12, 1))
daily_close_px = all_data['Adj. Close'].reset_index().pivot('Date', 'Ticker', 'Adj. Close')
all_data.to_csv('/Volumes/jannis/python/examples/aapl-msft-ibm-googl.csv')
daily_close_px.head()
daily_close_px.fillna(0, inplace = True)
instrumentIds = ['AAPL', 'MSFT', 'IBM', 'GOOGL', 'FB', 'EBAY', 'AMD', 'ADBE']
scores, pvalues, pairs = find_cointegrated_pairs(daily_close_px)

import seaborn
m = [0, 0.2, 0.4, 0.6, 0.8, 1]
seaborn.heatmap(pvalues, xticklabels = instrumentIds, yticklabels = instrumentIds, 
                cmap = 'RdYlGn_r', mask = (pvalues >= 0.98))
plt.show()
print(pairs)

s1 = daily_close_px['WIKI/ADBE']#.plot(figsize = (15,7))
s2 = daily_close_px['WIKI/MSFT']#.plot()
#plt.legend(['Adobe', 'Microsoft'])
#plt.show()

score, pvalue, _ = coint(s1, s2)
print(pvalue)
ratios = s1 / s2
ratios.plot(figsize = (15, 7))
plt.axhline(ratios.mean())
plt.legend(['Ratio'])
plt.show()

def zscore(series):
    return (series - series.mean()) / np.std(series)
    
zscore(ratios).plot()
plt.axhline(zscore(ratios).mean())
plt.axhline(1.0, color = 'red')
plt.axhline(-1.0, color = 'green')
plt.show()

print(len(ratios))

train = ratios[:2000]
test = ratios[2000:]

ratios_mavg5 = train.rolling(window=5, center = False).mean()
ratios_mavg60 = train.rolling(window=60, center = False).mean()
std_60 = train.rolling(window=60, center = False).std()
zscore_60_5 = (ratios_mavg5 - ratios_mavg60) / std_60

plt.figure(figsize = (15,7))
plt.plot(train.index, train.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg60.index, ratios_mavg60.values)
plt.legend(['Ratio', '5d Ratio MA', '60d Ratio MA'])

plt.ylabel('Ratio')
plt.show()

plt.figure(figsize = (15,7))
zscore_60_5.plot()
plt.axhline(0, color = 'black')
plt.axhline(1.0, color = 'red', linestyle = '--')
plt.axhline(-1.0, color = 'green', linestyle = '--')
plt.legend(['Rolling Ratio z-score', 'Mean', '+1', '-1'])
plt.show()

# train & validate

plt.figure(figsize = (15, 7))

train[60:].plot()
buy = train.copy()
sell = train.copy()
buy[zscore_60_5 > -1] = 0
sell[zscore_60_5 < 1] = 0
buy[60:].plot(color = 'g', linestyle = 'None', marker = '^')
sell[60:].plot(color = 'r', linestyle = 'None', marker = 'v')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, ratios.min(), ratios.max()))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()

# plot prices and buy and sell signals from z-score
plt.figure(figsize = (18,9))
s1 = daily_close_px['WIKI/ADBE'].iloc[:2000]
s2 = daily_close_px['WIKI/MSFT'].iloc[:2000]

s1[60:].plot(color = 'b')
s2[60:].plot(color = 'c')
buyR = 0*s1.copy()
sellR = 0*s1.copy()

# buy the ratio, buy s1 and sell s2

buyR[buy != 0] = s1[buy != 0]
sellR[buy != 0] = s2[buy != 0]

# sell ratio, sell s1 and buy s2

buyR[sell != 0] = s2[sell !=0]
sellR[sell != 0] = s1[sell != 0]

buyR[60:].plot(color = 'g', linestyle = 'None', marker = '^')
sellR[60:].plot(color = 'r', linestyle = 'None', marker = 'v')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, min(s1.min(), s2.min()), max(s1.max(), s2.max())))

plt.legend(['ADBE', 'MSFT', 'Buy Signal', 'Sell Signal'])
plt.show()

# trade strategy

def trade(s1, s2, window1, window2):
    if (window1 == 0) or (window2 == 0):
        return 0
    
    ratios = s1/s2
    ma1 = ratios.rolling(window = window1, center = False).mean()
    ma2 = ratios.rolling(window = window2, center = False).mean()
    std = ratios.rolling(window = window1, center = False).std()
    zscore = (ma1 - ma2)/std

    money = 0
    counts1 = 0
    counts2 = 0
    for i in range(len(ratios)):
        # sell short if z-score is > 1
        if zscore[i] >1:
            money += s1[i] - s2[i]*ratios[i]
            counts1 -= 1
            counts2 += ratios[i]
            print('Selling Ratio %s %s %s %s '%(money, ratios[i], counts1, counts2))
        # buy long if z-score is < 1
        elif zscore[i] < -1:
            money -= s1[i] - s2[i]*ratios[i]
            counts1 += 1
            counts2 -= ratios[i]
            print('Buying Ratio %s %s %s %s '%(money, ratios[i], counts1, counts2))
        # clear positions if z-score is between -.5 and .5
        elif abs(zscore[i]) < 0.75:
            money += s1[i] * counts1 + s2[i] * counts2
            counts1 = 0
            counts2 = 0
            print('Exit pos %s %s %s %s' %(money, ratios[i], counts1, counts2))
    return money
# run trades on train data    
s1_train = daily_close_px['WIKI/ADBE'].iloc[:2000]
s2_train = daily_close_px['WIKI/MSFT'].iloc[:2000]

trade(s1_train, s2_train, 5, 60)

# backtest on test data

s1_test = daily_close_px['WIKI/ADBE'].iloc[2000:]
s2_test = daily_close_px['WIKI/MSFT'].iloc[2000:]

trade(s1_test, s2_test, 5, 60)

# find window length that gives highest returns in this strategy

length_scores = [trade(s1_train, s2_train, 5, l) for l in range(255)]
best_length = np.argmax(length_scores)
print('Best window length: ', best_length)                 

# find window length on test data, compare returns
length_scores2 = [trade(s1_test, s2_test, 5, l) for l in range(255)]
print(best_length, 'day window: ', length_scores2[best_length])

best_length2 = np.argmax(length_scores2)
print(best_length2, 'day window: ', length_scores2[best_length2])            

plt.figure(figsize = (15, 7))
plt.plot(length_scores)
plt.plot(length_scores2)
plt.xlabel('Window length')
plt.ylabel('Score')
plt.legend(['Training', 'Test'])
plt.show()