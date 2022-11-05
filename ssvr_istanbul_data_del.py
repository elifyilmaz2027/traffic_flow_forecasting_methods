from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pandas as pd


def gridsearch(train_X, train_y, parameters):
    svr = SVR()
    grid_search = GridSearchCV(svr, parameters, cv=2)
    grid_search.fit(train_X, train_y)
    best_parameters = grid_search.best_params_
    return best_parameters


def ssvr_model(train_X, train_y, test_X, best_parameters):
    C = best_parameters['C']
    kernel = best_parameters['kernel']
    svr = SVR(kernel=kernel, C=C)
    svr.fit(train_X, train_y)
    train_predictions = svr.predict(train_X)
    test_predictions = svr.predict(test_X)
    pd.DataFrame(train_predictions).to_csv("point_forecasts/ssvr_istanbul_data_del_train.csv")
    pd.DataFrame(test_predictions).to_csv("point_forecasts/ssvr_istanbul_data_del_test.csv")
    return train_predictions, test_predictions

data = pd.read_csv('data/istanbul/istanbul_data_del.csv')[['NUMBER_OF_VEHICLES']]
data2 = pd.DataFrame(data.values)
data3 = pd.concat([data2.shift(169),data2.shift(168),data2.shift(167),data2.shift(25),data2.shift(24),data2.shift(23),data2.shift(2),data2.shift(1),data2], axis=1)
data3.columns = ['t-169','t-168','t-167','t-25','t-24','t-23','t-2','t-1', 't']

data4 = data3.values
train_size4 = 7565
train, test = data4[169:train_size4], data4[train_size4:]
train_X, train_y = train[:,:8], train[:,-1]
test_X, test_y = test[:,:8], test[:,-1]

parameters = {'kernel':['rbf', 'linear'], 'C':[0.1, 1, 10, 100]}
best_parameters = gridsearch(train_X, train_y, parameters)
ssvr_model(train_X, train_y, test_X, best_parameters)