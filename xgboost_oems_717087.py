from numpy import asarray
from xgboost import XGBRegressor
import pandas as pd
import numpy as np


 
"""# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	agg = concat(cols, axis=1)
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values"""
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]
 
# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
	train = asarray(train)
	trainX, trainy = train[:, :-1], train[:, -1]
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	yhat = model.predict(testX)
	return yhat
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    train_predictions = []
    test_predictions = []
    train, test = train_test_split(data, n_test)
    history = [x for x in train]
    train = asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    testX, testy = test[:, :-1], test[:, -1]
    yhat_train = xgboost_forecast(history, trainX)
    train_predictions.append(yhat_train)
    yhat_test = xgboost_forecast(history, testX)
    test_predictions.append(yhat_test)
    return train_predictions, test_predictions
 

data = pd.read_csv('data/pems/pems-d07-9months-2021-station717087-15min.csv')[['Total Flow']]
data2 = pd.DataFrame(data.values)
data3 = pd.concat([data2.shift(673),data2.shift(672),data2.shift(671),data2.shift(97),data2.shift(96),data2.shift(95),data2.shift(2),data2.shift(1),data2], axis=1)
data3.columns = ['t-673','t-672','t-671','t-97','t-96','t-95','t-2','t-1','t']
train_size = 17672
test_size = 7863

values = data3.values
data = values[673:,:]
train_predictions, test_predictions = walk_forward_validation(data, 7863)
train_predictions = pd.DataFrame(np.array(train_predictions).reshape(train_size,1))
test_predictions = pd.DataFrame(np.array(test_predictions).reshape(test_size,1))
train_predictions.to_csv("point_forecasts/xgboost_pems_717087_train.csv")
test_predictions.to_csv("point_forecasts/xgboost_pems_717087_test.csv")

