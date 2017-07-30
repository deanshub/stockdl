# LSTM for international airline passengers problem with regression framing
import os
import numpy
import datetime
import calendar
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas
import math
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def get_stock_data(stock_name, normalized=0):
	today = datetime.date.today()
	filename = stock_name+'_stock_%s.csv' % today

	if not os.path.isfile(filename):
		month = calendar.month_abbr[7]
		day = '17'
		year='2012'
		url = 'http://www.google.com/finance/historical?q={0}&startdate={1}+{2}%2C+{3}&output=csv'.format(stock_name, month, day,year)

		col_names = ['Date','Open','High','Low','Close','Volume']
		# stocks = pd.read_csv(url, header=0, names=col_names)
		stocks = pandas.read_csv(url, header=1, names=col_names)
		df = pandas.DataFrame(stocks)
		# date_split = df['Date'].str.split('-').str
		# df['Year'], df['Month'], df['Day'] = date_split
		# df["Volume"] = df["Volume"] / 10000
		# df.drop(df.columns[[0,3,5,6, 7,8,9]], axis=1, inplace=True)
		df.drop(df.columns[[0,3,5]], axis=1, inplace=True)
		df.to_csv(filename,index=True)
	return filename

def writeModel(model, filename):
	model.save(filename)
	# jsonModel = model.to_json()
	# file = open(filename,"w")
	# file.write(jsonModel)
	# file.close()

def readModel(filename):
	return load_model(filename)
	# file = open(filename,"r")
	# jsonModel = file.read()
	# file.close()
	# return model_from_json(jsonModel)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)


def trainModel(filename, look_back, trainX, trainY):
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
	writeModel(model, filename+'.h5')
	return model

def getModel(filename, look_back, trainX, trainY):
	if os.path.isfile(filename+'.h5'):
		model = readModel(filename+'.h5')
	else:
		model = trainModel(filename, look_back, trainX, trainY)
	return model

def getPrediciton(dataset, look_back):
	dataset = scaler.fit_transform(dataset)
	# split into train and test sets
	train_size = int(len(dataset))
	test_size = 1
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

	# futurePrediction = numpy.zeros((int(len(dataset) * 1.3),1))
	# futurePrediction[:dataset.shape[0]] = dataset


	# reshape into X=t and Y=t+1
	trainX, trainY = create_dataset(train, look_back)
	# testX, testY = create_dataset(test, look_back)

	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	model = getModel(filename, look_back, trainX, trainY)


	# make predictions
	# print(testX)
	trainPredict = model.predict(trainX)
	# testPredict = model.predict(testX)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	return trainPredict

stockSymbol = 'FATE'
daysToAdd = 50
filename = get_stock_data(stockSymbol)
# load the dataset
# dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataframe = read_csv(filename, usecols=[3], engine='python', skipfooter=3)
dataset = dataframe.values[::-1]
dataset = dataset.astype('float32')
lastActualValue = dataset[len(dataset) - 1]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
look_back = 1

trainPredict = getPrediciton(dataset, look_back)
for x in range(0,daysToAdd):
	print(trainPredict[len(trainPredict)-1])
	dataset = numpy.vstack([dataset, [trainPredict[len(trainPredict)-1]]])
	trainPredict = getPrediciton(dataset, look_back)

# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])

# calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

lastPredictionValue = trainPredict[len(trainPredict)-1]
precentageDiff = numpy.absolute(lastActualValue - lastPredictionValue) / ((lastActualValue + lastPredictionValue)/2)
precentageDiff=precentageDiff*100
sign = '+' if (lastPredictionValue>lastActualValue) else '-'

predictionDate = datetime.datetime.now() + timedelta(days=daysToAdd)
predictionDate = predictionDate.strftime('%d - %m - %Y')
# plot baseline and predictions
plt.title(predictionDate + '  ' + stockSymbol+' - ' + lastPredictionValue + '('+ sign + precentageDiff+'%)')
# plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
plt.show()
