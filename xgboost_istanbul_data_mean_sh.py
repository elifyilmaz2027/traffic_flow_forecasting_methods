from numpy import asarray
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

 
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
 

data = pd.read_csv('data/istanbul/istanbul_data_mean_sh.csv')[['NUMBER_OF_VEHICLES']]
train_size = 7995
test_size = 3500
data2 = pd.DataFrame(data.values)
data3 = pd.concat([data2.shift(169),data2.shift(168),data2.shift(167),data2.shift(25),data2.shift(24),data2.shift(23),data2.shift(2),data2.shift(1),data2], axis=1)
data3.columns = ['t-169','t-168','t-167','t-25','t-24','t-23','t-2','t-1', 't']

values = data3.values
data = values[169:,:]
train_predictions, test_predictions = walk_forward_validation(data, 3500)
train_predictions = pd.DataFrame(np.array(train_predictions).reshape(train_size,1))
test_predictions = pd.DataFrame(np.array(test_predictions).reshape(test_size,1))
train_predictions.to_csv("point_forecasts/xgboost_istanbul_data_mean_sh_train.csv")
test_predictions.to_csv("point_forecasts/xgboost_istanbul_data_mean_sh_test.csv")

