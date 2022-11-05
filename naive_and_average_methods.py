import pandas as pd
import numpy as np

def naive_method(data):
    data2 = pd.DataFrame(data.values)
    data3 = pd.concat([data2.shift(1), data2], axis=1)
    data3.columns = ['t-1', 't']
    data4 = data3.values
    train_size = int(len( data4) * 0.70)
    train, test =  data4[1:train_size],  data4[train_size:]
    test_predictions, actual_values = test[:,0], test[:,1]
    return test_predictions, actual_values

def average_method(data):
    data2 = pd.DataFrame(data.values)
    data3 = pd.concat([data2.shift(1), data2], axis=1)
    data3.columns = ['t-1', 't']
    avg_values = []
    for i in range(len(data3)):
        avg_value = np.mean(data3['t-1'][:(i+1)])
        avg_values.append(avg_value)
    avg_values = pd.DataFrame(avg_values)
    data4 = pd.concat([data3, avg_values], axis = 1)
    data4['avg'] = data4[0]
    data4 = data4[['avg', 't']]
    data4 = data4.values
    train_size = int(len( data4) * 0.70)
    train, test =  data4[1:train_size],  data4[train_size:]
    test_predictions, actual_values = test[:,0], test[:,1]
    return test_predictions, actual_values