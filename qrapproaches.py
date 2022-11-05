import numpy as np
import statsmodels.formula.api as smf
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

## historical PI implementation
def get_corridor(corridor_size, error_series):
    corridor = []
    length = len(error_series) - corridor_size
    for i in range(length):
        errors_in_corridor = error_series[i:i+corridor_size]
        ordered_errors_in_corridor = np.sort(errors_in_corridor)
        corridor.append({"OrderedErrors": ordered_errors_in_corridor})
    return corridor

def get_lower_upper_bounds(corridor_size, predictions, error_series):
    PIs = []
    corridor = get_corridor(corridor_size, error_series)
    predictions2 = np.array(predictions[corridor_size:])
    percent5_index = 1
    percent95_index = 19
    for i in range(len(corridor)):
        OrderedErrors = corridor[i]["OrderedErrors"]
        PointForecast = predictions2[i]
        lower_bound = OrderedErrors[percent5_index] + PointForecast
        upper_bound = OrderedErrors[percent95_index] + PointForecast
        PIs.append([lower_bound, upper_bound])
    return PIs

## implement distribution-based PI using AR
def distribution_based_PI(test_data, a, sigma, z_alpha, z_1minalpha):
    PIs = []
    for i in range(len(test_data)):
        lower_bound = (a * test_data[i]) + (sigma * z_alpha)
        upper_bound = (a * test_data[i]) + (sigma * z_1minalpha)
        PIs.append([lower_bound, upper_bound])
    return PIs

#implement QRA
def qra(train_dataframe, test_dataframe, tau1, tau2):
    #tau1=0.95
    #tau2=0.05
    #since the best 3 models are ssvr, slstm and sxgboost
    #we use these models in QRA.
    model1 = smf.quantreg('NUMBER_OF_VEHICLES ~ ssvr + slstm + sxgboost', train_dataframe).fit(q=tau1)
    get_y = lambda a, b, c, d: a + b * test_dataframe.ssvr + c * test_dataframe.slstm + d * test_dataframe.sxgboost
    y_upper = get_y(model1.params['Intercept'], model1.params['ssvr'], model1.params['slstm'], model1.params['sxgboost'])
    model2 = smf.quantreg('NUMBER_OF_VEHICLES ~ ssvr + slstm + sxgboost', train_dataframe).fit(q=tau2)
    y_lower = get_y(model2.params['Intercept'], model2.params['ssvr'], model2.params['slstm'], model1.params['sxgboost'])
    y_upper = np.array(y_upper)
    y_lower = np.array(y_lower)

    PIs_qra = []
    for i in range(len(y_upper)):
        PIs_qra.append([y_lower[i], y_upper[i]])
    return PIs_qra

#implement QRLSTM
#first implemnet lstm
class LSTM(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        _, (h_output, _) = self.lstm(x, (h_0, c_0))
        h_output = h_output.view(-1, self.hidden_size)
        output = self.softmax(h_output)
        output = self.fc(output)
        return output

#define pinball loss to use update parameters in lstm
class PinballLoss():
    def __init__(self, quantile=0.10, reduction='mean'):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction
    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target 
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1-self.quantile) * (abs(error)[bigger_index])
        
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss

def qrlstm(data):
    sc = MinMaxScaler()
    training_data = sc.fit_transform(data)

    x, y = training_data[169:,:8], training_data[169:,-1]
    print(x.shape)
    print(y.shape)
    train_size = 7995

    dataX = Variable(torch.Tensor(np.array(x))).reshape((11495,8,1))
    dataY = Variable(torch.Tensor(np.array(y))).reshape((11495,1))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size]))).reshape((7995,8,1))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size]))).reshape((7995,1))

    testX = Variable(torch.Tensor(np.array(x[train_size:]))).reshape((3500,8,1))
    testY = Variable(torch.Tensor(np.array(y[train_size:]))).reshape((3500,1))
    num_epochs = 2000
    learning_rate = 0.01

    input_size = 1
    hidden_size = 8
    num_layers = 1
    num_classes = 1

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    criterion = PinballLoss(quantile=0.95)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, trainY)
        
        loss.backward()
        
        optimizer.step()
    
    lstm.eval()
    test_predict = lstm(testX)

    data_predict = test_predict.data.numpy()
    dataY_plot = dataY.data.numpy()
    
    d_p = np.concatenate((data_predict,data_predict,data_predict,data_predict,data_predict,data_predict,data_predict,data_predict,data_predict),axis=1)
    dY_p = np.concatenate((dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot),axis=1)
    data_predict = sc.inverse_transform(d_p)
    dataY_plot = sc.inverse_transform(dY_p)

    dataY_plot = dataY_plot[:,0]
    data_predict = data_predict[:,0]
    upper_bounds = data_predict
    
    #get lower bounds
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    criterion = PinballLoss(quantile=0.05)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs, trainY)
        
        loss.backward()
        
        optimizer.step()
    
    lstm.eval()
    test_predict = lstm(testX)
    data_predict = test_predict.data.numpy()
    dataY_plot = dataY.data.numpy()

    d_p = np.concatenate((data_predict,data_predict,data_predict,data_predict,data_predict,data_predict,data_predict,data_predict,data_predict),axis=1)
    dY_p = np.concatenate((dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot),axis=1)
    data_predict = sc.inverse_transform(d_p)
    dataY_plot = sc.inverse_transform(dY_p)

    dataY_plot = dataY_plot[:,0]
    data_predict = data_predict[:,0]
    lower_bounds = data_predict
    
    y_upper = np.array(upper_bounds)
    y_lower = np.array(lower_bounds)

    PIs_qrlstm = []
    for i in range(len(y_upper)):
        PIs_qrlstm.append([y_lower[i], y_upper[i]])
    return PIs_qrlstm
        
    
