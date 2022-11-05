import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

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
        output= self.fc(output)
        return output

data = pd.read_csv('data/istanbul/istanbul_data_mean_sh.csv')[['NUMBER_OF_VEHICLES']]
data2 = pd.DataFrame(data.values)
data3 = pd.concat([data2.shift(169),data2.shift(168),data2.shift(167),data2.shift(25),data2.shift(24),data2.shift(23),data2.shift(2),data2.shift(1),data2], axis=1)
data3.columns = ['t-169','t-168','t-167','t-25','t-24','t-23','t-2','t-1', 't']
data4 = data3.values
train_size = 7995
test_size = 3500

minmaxscaler = MinMaxScaler()
training_data = minmaxscaler.fit_transform(data4)
x, y = training_data[169:,:8], training_data[169:,-1]

dataX = Variable(torch.Tensor(np.array(x))).reshape((len(x),8,1))
dataY = Variable(torch.Tensor(np.array(y))).reshape((len(x),1))
trainX = Variable(torch.Tensor(np.array(x[0:train_size]))).reshape((train_size,8,1))
trainY = Variable(torch.Tensor(np.array(y[0:train_size]))).reshape((train_size,1))
testX = Variable(torch.Tensor(np.array(x[train_size:]))).reshape((test_size,8,1))
testY = Variable(torch.Tensor(np.array(y[train_size:]))).reshape((test_size,1))

num_epochs = 2000
learning_rate = 0.01

input_size = 1
hidden_size = 8
num_layers = 1
num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    # obtain the loss function
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
      
lstm.eval()
train_predict = lstm(trainX)
data_predict2 = train_predict.data.numpy()
d_p2 = np.concatenate((data_predict2,data_predict2,data_predict2,data_predict2,data_predict2,data_predict2,data_predict2,data_predict2,data_predict2),axis=1)
data_predict2 = minmaxscaler.inverse_transform(d_p2)
train_predict = data_predict2[:,0]

lstm.eval()
test_predict = lstm(testX)
data_predict = test_predict.data.numpy()
dataY_plot = dataY.data.numpy()
d_p = np.concatenate((data_predict,data_predict,data_predict,data_predict,data_predict,data_predict,data_predict,data_predict,data_predict),axis=1)
dY_p = np.concatenate((dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot,dataY_plot),axis=1)
data_predict = minmaxscaler.inverse_transform(d_p)
dataY_plot = minmaxscaler.inverse_transform(dY_p)
dataY_plot = dataY_plot[:,0]
test_predict = data_predict[:,0]


pd.DataFrame(train_predict).to_csv("point_forecasts/slstm_istanbul_data_mean_sh_train.csv")
pd.DataFrame(test_predict).to_csv("point_forecasts/slstm_istanbul_data_mean_sh_test.csv")

