import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd

def arimamodel(train_data, test_data):
    arima = ARIMA(train_data, order=(1,1,0))
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
    return test_predictions

def hybrid_model(train_predictions, train_data, test_predictions, test_data):
    train_data = np.array(train_data).reshape((len(train_data),1))
    train_predictions = np.array(train_predictions).reshape((len(train_predictions),1))
    test_data = np.array(test_data).reshape((len(test_data),1))
    test_predictions = np.array(test_predictions).reshape((len(test_predictions),1))
    train_error_series = train_data - train_predictions
    test_error_series = test_data - test_predictions
    #model residuals
    testerror_predictions = arimamodel(train_error_series, test_error_series)
    testerror_predictions = np.array(testerror_predictions).reshape((len(testerror_predictions),1))
    output = test_predictions + testerror_predictions
    output = np.array(output).reshape((len(output),1))
    return output

traindata_716933 = np.array(pd.read_csv('data/pems/pems-d07-9months-2021-station716933-15min.csv')[['Total Flow']][673:18345])
testdata_716933 = np.array(pd.read_csv('data/pems/pems-d07-9months-2021-station716933-15min.csv')[['Total Flow']][18345:])
trainslstm_716933 = np.array(pd.read_csv("point_forecasts/slstm_pems_716933_train.csv")["0"])
trainssvr_716933 = np.array(pd.read_csv("point_forecasts/ssvr_pems_716933_train.csv")["0"])
trainsxgboost_716933 = np.array(pd.read_csv("point_forecasts/sxgboost_pems_716933_train.csv")["0"])
testslstm_716933 = np.array(pd.read_csv("point_forecasts/slstm_pems_716933_test.csv")["0"])
testssvr_716933 = np.array(pd.read_csv("point_forecasts/ssvr_pems_716933_test.csv")["0"])
testsxgboost_716933 = np.array(pd.read_csv("point_forecasts/sxgboost_pems_716933_test.csv")["0"])

slstmarima_716933 = hybrid_model(trainslstm_716933, traindata_716933, testslstm_716933, testdata_716933)
ssvrarima_716933 = hybrid_model(trainssvr_716933, traindata_716933, testssvr_716933, testdata_716933)
sxgboostarima_716933 = hybrid_model(trainsxgboost_716933, traindata_716933, testsxgboost_716933, testdata_716933)
pd.DataFrame(slstmarima_716933).to_csv("point_forecasts/slstmarima_pems_716933_test.csv")
pd.DataFrame(ssvrarima_716933).to_csv("point_forecasts/ssvrarima_pems_716933_test.csv")
pd.DataFrame(sxgboostarima_716933).to_csv("point_forecasts/sxgboostarima_pems_716933_test.csv")

traindata_717087 = np.array(pd.read_csv('data/pems/pems-d07-9months-2021-station717087-15min.csv')[['Total Flow']][673:18345])
testdata_717087 = np.array(pd.read_csv('data/pems/pems-d07-9months-2021-station717087-15min.csv')[['Total Flow']][18345:])
trainslstm_717087 = np.array(pd.read_csv("point_forecasts/slstm_pems_717087_train.csv")["0"])
trainssvr_717087 = np.array(pd.read_csv("point_forecasts/ssvr_pems_717087_train.csv")["0"])
trainsxgboost_717087 = np.array(pd.read_csv("point_forecasts/sxgboost_pems_717087_train.csv")["0"])
testslstm_717087 = np.array(pd.read_csv("point_forecasts/slstm_pems_717087_test.csv")["0"])
testssvr_717087 = np.array(pd.read_csv("point_forecasts/ssvr_pems_717087_test.csv")["0"])
testsxgboost_717087 = np.array(pd.read_csv("point_forecasts/sxgboost_pems_717087_test.csv")["0"])

slstmarima_717087 = hybrid_model(trainslstm_717087, traindata_717087, testslstm_717087, testdata_717087)
ssvrarima_717087 = hybrid_model(trainssvr_717087, traindata_717087, testssvr_717087, testdata_717087)
sxgboostarima_717087 = hybrid_model(trainsxgboost_717087, traindata_717087, testsxgboost_717087, testdata_717087)
pd.DataFrame(slstmarima_717087).to_csv("point_forecasts/slstmarima_pems_717087_test.csv")
pd.DataFrame(ssvrarima_717087).to_csv("point_forecasts/ssvrarima_pems_717087_test.csv")
pd.DataFrame(sxgboostarima_717087).to_csv("point_forecasts/sxgboostarima_pems_717087_test.csv")
