import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd

def armodel(train_data, test_data):
    arima = ARIMA(train_data, order=(1,0,0))
    arima_fit = arima.fit()    
    print(arima_fit.summary())
    parameters = arima_fit.params
    a = parameters[1]
    output_test = arima_fit.forecast()
    test_predictions = []
    test_predictions.append(output_test[0][0])
    for t in range(len(test_data)-1):
        output_test = (test_data[t] * a)
        test_predictions.append(output_test[0])
    return test_predictions

def hybrid_model(train_predictions, train_data, test_predictions, test_data):
    train_data = np.array(train_data).reshape((len(train_data),1))
    train_predictions = np.array(train_predictions).reshape((len(train_predictions),1))
    test_data = np.array(test_data).reshape((len(test_data),1))
    test_predictions = np.array(test_predictions).reshape((len(test_predictions),1))
    train_error_series = train_data - train_predictions
    test_error_series = test_data - test_predictions
    #model residuals
    testerror_predictions = armodel(train_error_series, test_error_series)
    testerror_predictions = np.array(testerror_predictions).reshape((len(testerror_predictions),1))
    output = test_predictions + testerror_predictions
    output = np.array(output).reshape((len(output),1))
    return output

traindata_istanbul_data_del = np.array(pd.read_csv('data/istanbul/istanbul_data_del.csv')[['NUMBER_OF_VEHICLES']][169:7565])
testdata_istanbul_data_del = np.array(pd.read_csv('data/istanbul/istanbul_data_del.csv')[['NUMBER_OF_VEHICLES']][7565:])
trainslstm_istanbul_data_del = np.array(pd.read_csv("point_forecasts/slstm_istanbul_data_del_train.csv")["0"])
trainssvr_istanbul_data_del = np.array(pd.read_csv("point_forecasts/ssvr_istanbul_data_del_train.csv")["0"])
trainsxgboost_istanbul_data_del = np.array(pd.read_csv("point_forecasts/sxgboost_istanbul_data_del_train.csv")["0"])
testslstm_istanbul_data_del = np.array(pd.read_csv("point_forecasts/slstm_istanbul_data_del_test.csv")["0"])
testssvr_istanbul_data_del = np.array(pd.read_csv("point_forecasts/ssvr_istanbul_data_del_test.csv")["0"])
testsxgboost_istanbul_data_del = np.array(pd.read_csv("point_forecasts/sxgboost_istanbul_data_del_test.csv")["0"])

slstmarima_istanbul_data_del = hybrid_model(trainslstm_istanbul_data_del, traindata_istanbul_data_del, testslstm_istanbul_data_del, testdata_istanbul_data_del)
ssvrarima_istanbul_data_del = hybrid_model(trainssvr_istanbul_data_del, traindata_istanbul_data_del, testssvr_istanbul_data_del, testdata_istanbul_data_del)
sxgboostarima_istanbul_data_del = hybrid_model(trainsxgboost_istanbul_data_del, traindata_istanbul_data_del, testsxgboost_istanbul_data_del, testdata_istanbul_data_del)
pd.DataFrame(slstmarima_istanbul_data_del).to_csv("point_forecasts/slstmarima_istanbul_data_del_test.csv")
pd.DataFrame(ssvrarima_istanbul_data_del).to_csv("point_forecasts/ssvrarima_istanbul_data_del_test.csv")
pd.DataFrame(sxgboostarima_istanbul_data_del).to_csv("point_forecasts/sxgboostarima_istanbul_data_del_test.csv")

traindata_istanbul_data_mean_sh = np.array(pd.read_csv('data/istanbul/istanbul_data_mean_sh.csv')[['NUMBER_OF_VEHICLES']][169:8164])
testdata_istanbul_data_mean_sh = np.array(pd.read_csv('data/istanbul/istanbul_data_mean_sh.csv')[['NUMBER_OF_VEHICLES']][8164:])
trainslstm_istanbul_data_mean_sh = np.array(pd.read_csv("point_forecasts/slstm_istanbul_data_mean_sh_train.csv")["0"])
trainssvr_istanbul_data_mean_sh = np.array(pd.read_csv("point_forecasts/ssvr_istanbul_data_mean_sh_train.csv")["0"])
trainsxgboost_istanbul_data_mean_sh = np.array(pd.read_csv("point_forecasts/sxgboost_istanbul_data_mean_sh_train.csv")["0"])
testslstm_istanbul_data_mean_sh = np.array(pd.read_csv("point_forecasts/slstm_istanbul_data_mean_sh_test.csv")["0"])
testssvr_istanbul_data_mean_sh = np.array(pd.read_csv("point_forecasts/ssvr_istanbul_data_mean_sh_test.csv")["0"])
testsxgboost_istanbul_data_mean_sh = np.array(pd.read_csv("point_forecasts/sxgboost_istanbul_data_mean_sh_test.csv")["0"])

slstmarima_istanbul_data_mean_sh = hybrid_model(trainslstm_istanbul_data_mean_sh, traindata_istanbul_data_mean_sh, testslstm_istanbul_data_mean_sh, testdata_istanbul_data_mean_sh)
ssvrarima_istanbul_data_mean_sh = hybrid_model(trainssvr_istanbul_data_mean_sh, traindata_istanbul_data_mean_sh, testssvr_istanbul_data_mean_sh, testdata_istanbul_data_mean_sh)
sxgboostarima_istanbul_data_mean_sh = hybrid_model(trainsxgboost_istanbul_data_mean_sh, traindata_istanbul_data_mean_sh, testsxgboost_istanbul_data_mean_sh, testdata_istanbul_data_mean_sh)
pd.DataFrame(slstmarima_istanbul_data_mean_sh).to_csv("point_forecasts/slstmarima_istanbul_data_mean_sh_test.csv")
pd.DataFrame(ssvrarima_istanbul_data_mean_sh).to_csv("point_forecasts/ssvrarima_istanbul_data_mean_sh_test.csv")
pd.DataFrame(sxgboostarima_istanbul_data_mean_sh).to_csv("point_forecasts/sxgboostarima_istanbul_data_mean_sh_test.csv")


traindata_istanbul_data_mean_sdsh= np.array(pd.read_csv('data/istanbul/istanbul_data_mean_sdsh.csv')[['NUMBER_OF_VEHICLES']][169:8164])
testdata_istanbul_data_mean_sdsh= np.array(pd.read_csv('data/istanbul/istanbul_data_mean_sdsh.csv')[['NUMBER_OF_VEHICLES']][8164:])
trainslstm_istanbul_data_mean_sdsh= np.array(pd.read_csv("point_forecasts/slstm_istanbul_data_mean_sdsh_train.csv")["0"])
trainssvr_istanbul_data_mean_sdsh= np.array(pd.read_csv("point_forecasts/ssvr_istanbul_data_mean_sdsh_train.csv")["0"])
trainsxgboost_istanbul_data_mean_sdsh= np.array(pd.read_csv("point_forecasts/sxgboost_istanbul_data_mean_sdsh_train.csv")["0"])
testslstm_istanbul_data_mean_sdsh= np.array(pd.read_csv("point_forecasts/slstm_istanbul_data_mean_sdsh_test.csv")["0"])
testssvr_istanbul_data_mean_sdsh= np.array(pd.read_csv("point_forecasts/ssvr_istanbul_data_mean_sdsh_test.csv")["0"])
testsxgboost_istanbul_data_mean_sdsh= np.array(pd.read_csv("point_forecasts/sxgboost_istanbul_data_mean_sdsh_test.csv")["0"])

slstmarima_istanbul_data_mean_sdsh= hybrid_model(trainslstm_istanbul_data_mean_sdsh, traindata_istanbul_data_mean_sdsh, testslstm_istanbul_data_mean_sdsh, testdata_istanbul_data_mean_sdsh)
ssvrarima_istanbul_data_mean_sdsh= hybrid_model(trainssvr_istanbul_data_mean_sdsh, traindata_istanbul_data_mean_sdsh, testssvr_istanbul_data_mean_sdsh, testdata_istanbul_data_mean_sdsh)
sxgboostarima_istanbul_data_mean_sdsh= hybrid_model(trainsxgboost_istanbul_data_mean_sdsh, traindata_istanbul_data_mean_sdsh, testsxgboost_istanbul_data_mean_sdsh, testdata_istanbul_data_mean_sdsh)
pd.DataFrame(slstmarima_istanbul_data_mean_sdsh).to_csv("point_forecasts/slstmarima_istanbul_data_mean_sdsh_test.csv")
pd.DataFrame(ssvrarima_istanbul_data_mean_sdsh).to_csv("point_forecasts/ssvrarima_istanbul_data_mean_sdsh_test.csv")
pd.DataFrame(sxgboostarima_istanbul_data_mean_sdsh).to_csv("point_forecasts/sxgboostarima_istanbul_data_mean_sdsh_test.csv")
