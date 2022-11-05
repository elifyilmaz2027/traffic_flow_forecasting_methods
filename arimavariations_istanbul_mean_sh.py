from statsmodels.tsa.arima_model import ARIMA
import pandas as pd

def armodel(train_data, test_data):
    arima = ARIMA(train_data, order=(5,0,0))
    arima_fit = arima.fit()    
    print(arima_fit.summary())
    parameters = arima_fit.params
    a1 = parameters[1]
    a2 = parameters[2]
    a3 = parameters[3]
    a4 = parameters[4]
    a5 = parameters[5]
    train_predictions = []
    for t in range(4,len(train_data)):
        output_train = (train_data[t-4] * a5) + (train_data[t-3] * a4) + (train_data[t-2] * a3) + (train_data[t-1] * a2) + (train_data[t] * a1)
        train_predictions.append(output_train)
        
    test_data2=[]
    test_data2.append(train_data[-5])
    test_data2.append(train_data[-4])
    test_data2.append(train_data[-3])
    test_data2.append(train_data[-2])
    test_data2.append(train_data[-1])
    for i in range(len(test_data)-1):
        test_data2.append(test_data[i])
    
    test_predictions = []
    for t in range(4,len(test_data2)):
        output_test = (test_data2[t-4] * a5) + (test_data2[t-3] * a4) + (test_data2[t-2] * a3) + (test_data2[t-1] * a2) + (test_data2[t] * a1)
        test_predictions.append(output_test)
    pd.DataFrame(train_predictions).to_csv("point_forecasts/ar_istanbul_data_mean_sh_train.csv")
    pd.DataFrame(test_predictions).to_csv("point_forecasts/ar_istanbul_data_mean_sh_test.csv")
    return train_predictions, test_predictions

def armamodel(train_data, test_data):
    arima = ARIMA(train_data, order=(1,0,1))
    arima_fit = arima.fit()    
    print(arima_fit.summary())
    parameters = arima_fit.params
    a = parameters[1]
    b = parameters[2]
    output_train = arima_fit.forecast()
    train_predictions = []
    for t in range(len(train_data)):
        output_train = (train_data[t] * a) + ((train_data[t] - output_train[0]) * b)
        train_predictions.append(output_train[0])
        
    output_test = arima_fit.forecast()
    test_predictions = []
    test_predictions.append(output_test[0][0])
    for t in range(len(test_data)-1):
        output_test = (test_data[t] * a) + ((test_data[t] - output_test[0]) * b)
        test_predictions.append(output_test[0])
    pd.DataFrame(train_predictions).to_csv("point_forecasts/arma_istanbul_data_mean_sh_train.csv")
    pd.DataFrame(test_predictions).to_csv("point_forecasts/arma_istanbul_data_mean_sh_test.csv")
    return train_predictions, test_predictions


def arimamodel(train_data, test_data):
    arima = ARIMA(train_data, order=(0,1,3))
    arima_fit = arima.fit()    
    print(arima_fit.summary())
    
    train_predictions = arima_fit.predict(start=len(train_data),end=len(train_data)+len(train_data),dynamic=train_data.all())
    train_predictions2 = []
    for t in range(len(train_data)):
        output_train = train_predictions[t] + train_data[t]
        train_predictions2.append(output_train)
    
    test_predictions = arima_fit.predict(start=len(train_data),end=len(train_data)+len(test_data)-1,dynamic=test_data.all())
    test_predictions2 = []
    test_data2=[]
    test_data2.append(train_data[-1])
    for i in range(len(test_data)-1):
        test_data2.append(test_data[i])
    for t in range(len(test_data2)):
        output_test = test_predictions[t] + test_data2[t]
        test_predictions2.append(output_test)

    pd.DataFrame(train_predictions2).to_csv("point_forecasts/arima_istanbul_data_mean_sh_train.csv")
    pd.DataFrame(test_predictions2).to_csv("point_forecasts/arima_istanbul_data_mean_sh_test.csv")
    return train_predictions2, test_predictions2

def sarimamodel(data):
    data2 = pd.DataFrame(data)
    data3 = pd.concat([data2.shift(169),data2.shift(168),data2.shift(25),data2.shift(24),data2], axis=1)
    data3.columns = ['t-169','t-168','t-25','t-24','t']
    data4 = data3.values
    train_size = int(len(data4) * 0.70)
    train, test = data4[169:train_size], data4[train_size:]
    train_X, train_y = train[:,:4], train[:,-1]
    test_X, test_y = test[:,:4], test[:,-1]

    sarima = ARIMA(train_y, order=(1,1,2), exog=train_X)
    sarima_fit = sarima.fit()    
    print(sarima_fit.summary())
    
    train_predictions = sarima_fit.predict(start=len(train_y),end=len(train_y)+len(train_y)-1,dynamic=train_data.all(),exog=train_X)
    train_predictions2 = []
    for t in range(len(train_y)):
        output_train = train_predictions[t] + train_y[t]
        train_predictions2.append(output_train)
    
    test_predictions = sarima_fit.predict(start=len(train_y),end=len(train_y)+len(test_y)-1,dynamic=test_data.all(),exog=test_X)
    test_predictions2 = []
    test_y2=[]
    test_y2.append(train_y[-1])
    for i in range(len(test_y)-1):
        test_y2.append(test_y[i])
    for t in range(len(test_y2)):
        output_test = test_predictions[t] + test_y2[t]
        test_predictions2.append(output_test)

    pd.DataFrame(train_predictions2).to_csv("point_forecasts/sarima_istanbul_data_mean_sh_train.csv")
    pd.DataFrame(test_predictions2).to_csv("point_forecasts/sarima_istanbul_data_mean_sh_test.csv")
    return train_predictions2, test_predictions2


data = pd.read_csv('data/istanbul/istanbul_data_mean_sh.csv')[['NUMBER_OF_VEHICLES']]
data = data.values
train_size = int(len(data) * 0.70)
train_data, test_data = data[:train_size], data[train_size:]
armamodel(train_data, test_data)
armodel(train_data, test_data)
arimamodel(train_data, test_data)
sarimamodel(data)
