#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:56:22 2019

@author: ioannismilas
"""

# AR
from statsmodels.tsa.ar_model import AR
from random import random
import matplotlib.pyplot as plt

data = [x + random() for x in range(1, 100)]
plt.plot(data)
plt.show()

model = AR(data)
model_fit = model.fit()

yhat = model_fit.predict(len(data), len(data))
print(yhat)

#MA

from statsmodels.tsa.arima_model import ARMA

model = ARMA(data, order = (0,1))
model_fit = model.fit(disp= False)

yhat = model_fit.predict(len(data), len(data))
print(yhat)

#ARMA

model = ARMA(data, order=(2,1))
model_fit = model.fit(disp= False)

yhat = model_fit.predict(len(data), len(data))
print(yhat)

#ARIMA
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(data, order =(1,1,1))
model_fit = model.fit(disp=False)

yhat = model_fit.predict(len(data), len(data))
print(yhat)

# SARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

model= SARIMAX(data, order=(1,1,1,1), seasonal_order = (1,1,1,1))
model_fit = model.fit(disp=False)

yhat= model_fit.predict(len(data), len(data))
print(yhat)

# SARIMAX 

data1 = [x+ random() for x in range(1, 100)]
data2 = [x + random() for x in range(101, 200)]

model = SARIMAX(data1, exog=data2, order = (1,1,1), seasonal_order = (0, 0, 0, 0))
model_fit = model.fit(disp=False)

exog2 = [200 + random()]
yhat = model_fit.predict(len(data1), len(data1), exog = [exog2])
print(yhat)

# VAR

from statsmodels.tsa.vector_ar.var_model import VAR

data = list()
for i in range(100):
    v1 = i + random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)

model = VAR(data)
model_fit = model.fit()

yhat= model_fit.forecast(model_fit.y, steps=1)
print(yhat)

# VARMA

from statsmodels.tsa.statespace.varmax import VARMAX

model = VARMAX(data, order = (1,1))
model_fit = model.fit(disp=False)

yhat = model_fit.forecast()
print(yhat)

# VARMAX

data_exog = [x + random() for x in range(100)]

model = VARMAX(data, exog = data_exog, order = (1, 1))
model_fit = model.fit(disp=False)

data_exog2 = [[100]]
yhat= model_fit.forecast(exog=data_exog2)
print(yhat)

# simple exponential smoothing

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(data)
model_fit = model.fit()

yhat = model_fit.predict(len(data), len(data))
print(yhat)