from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return numpy.array(dataX), numpy.array(dataY)
#

data = read_csv('data.csv', usecols=[0], engine='python')
X = data.values

# cross validation
cv_params = numpy.arange(0.3, 0.7, 0.1)
for cv_param in cv_params:
    train_size = int(len(X) * cv_param)
    test_size = len(X) - train_size
    train, test = X[0:train_size], X[train_size:len(X)]
    # fit the model
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    history = [x for x in trainX]
    predictions = list()
    for t in range(len(test)):
	    model = ARIMA(history, order=(1,1,0))
	    model_fit = model.fit(disp=0)
	    output = model_fit.forecast()
	    yhat = output[0]
	    predictions.append(yhat)
	    obs = trainY[t]
	    history.append(obs)
