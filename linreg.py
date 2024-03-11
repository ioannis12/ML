# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:44:30 2019

@author: ioannismilas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

USAhousing = pd.read_csv('/Volumes/jannis/python/examples/machine-learning-master/USA_Housing.csv')
USAhousing.head()
USAhousing.info()
USAhousing.describe()
USAhousing.columns

sns.pairplot(USAhousing)
sns.distplot(USAhousing['Price'])
USAhousing.corr()
sns.heatmap(USAhousing.corr())
sns.plt.show()

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)


print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_,X.columns, columns=['Coefficient'])
print(coeff_df)

predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)

sns.distplot((y_test-predictions), bins=50)

from sklearn import metrics 
print('MAE:', metrics.mean_absolute_error(y_test, predictions))