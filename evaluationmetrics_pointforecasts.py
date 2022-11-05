import numpy as np

#Defining MAPE function
def MAPE(actual_values,predicted_values):
    predicted_values = np.array(predicted_values).reshape((len(predicted_values), 1))
    actual_values = np.array(actual_values).reshape((len(actual_values), 1))
    mape = np.mean(np.abs((actual_values - predicted_values)/actual_values))*100
    return mape

#Defining MAPE_100 function
def MAPE_100(actual_values,predicted_values):
    predicted_values = np.array(predicted_values).reshape((len(predicted_values), 1))
    actual_values = np.array(actual_values).reshape((len(actual_values), 1))
    x = np.concatenate((actual_values,predicted_values), axis=1)
    x_100 = x[x[:,0]>100]
    mape = np.mean(np.abs((x_100[:,0] - x_100[:,1]) / x_100[:,0]))*100
    return mape

#Defining MAPE_250 function
def MAPE_250(actual_values,predicted_values):
    predicted_values = np.array(predicted_values).reshape((len(predicted_values), 1))
    actual_values = np.array(actual_values).reshape((len(actual_values), 1))
    x = np.concatenate((actual_values,predicted_values), axis=1)
    x_250 = x[x[:,0]>250]
    mape = np.mean(np.abs((x_250[:,0] - x_250[:,1]) / x_250[:,0]))*100
    return mape

#Defining MAE function
def MAE(actual_values,predicted_values):
    predicted_values = np.array(predicted_values).reshape((len(predicted_values), 1))
    actual_values = np.array(actual_values).reshape((len(actual_values), 1))
    mae = np.mean(np.abs(actual_values - predicted_values))
    return mae