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
    pd.DataFrame(train_predictions).to_csv("point_forecasts/ssvr_pems_716933_train.csv")
    pd.DataFrame(test_predictions).to_csv("point_forecasts/ssvr_pems_716933_test.csv")
    return train_predictions, test_predictions

data = pd.read_csv('data/pems/pems-d07-9months-2021-station716933-15min.csv')[['Total Flow']]
data2 = pd.DataFrame(data.values)
data3 = pd.concat([data2.shift(673),data2.shift(672),data2.shift(671),data2.shift(97),data2.shift(96),data2.shift(95),data2.shift(2),data2.shift(1),data2], axis=1)
data3.columns = ['t-673','t-672','t-671','t-97','t-96','t-95','t-2','t-1','t']

data4 = data3.values
train_size4 = int(len(data4) * 0.70)
train, test = data4[673:train_size4], data4[train_size4:]
train_X, train_y = train[:,:8], train[:,-1]
test_X, test_y = test[:,:8], test[:,-1]

parameters = {'kernel':['rbf', 'linear'], 'C':[0.1, 1, 10, 100]}
best_parameters = gridsearch(train_X, train_y, parameters)
ssvr_model(train_X, train_y, test_X, best_parameters)