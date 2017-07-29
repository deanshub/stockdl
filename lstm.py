import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import math
import os

def get_stock_data(stock_name, normalized=0):
    # url = 'http://chart.finance.yahoo.com/table.csv?s=%s&a=11&b=15&c=2011&d=29&e=10&f=2016&g=d&ignore=.csv' % stock_name
    # url = 'http://download.finance.yahoo.com/d/quotes.csv?s=%s&a=11&b=15&c=2011&d=29&e=10&f=2016&g=d&ignore=.csv' % stock_name
    # url = 'http://download.finance.yahoo.com/d/quotes.csv?s=%s&f=d2aohgpv' % stock_name
    url = 'http://www.google.com/finance/historical?q={0}&startdate=Jan+01%2C+2013&output=csv'.format(stock_name)
    print(url)
    # col_names = ['Date','Open','High','Low','Close','Volume','Adj Close']
    col_names = ['Date','Open','High','Low','Close','Volume']
    # stocks = pd.read_csv(url, header=0, names=col_names)
    stocks = pd.read_csv(url, header=1, names=col_names)
    df = pd.DataFrame(stocks)
    date_split = df['Date'].str.split('-').str
    df['Year'], df['Month'], df['Day'] = date_split
    # df["Volume"] = df["Volume"] / 10000
    # df["High"] = df["High"] / 10000
    # df["Close"] = df["Close"] / 10000
    df = df[(df["Volume"]!="-")]
    print(df["Volume"].max(), (len(str(df["Volume"].max()))), pow(10,(len(str(df["Volume"].max())))))
    volumeMechane = pow(10,(len(str(df["Volume"].max()))))/10000
    print(volumeMechane)
    moneyMechane = pow(10,(len(str(df["High"].max()))))/10000
    df["Volume"] = [int(i)/volumeMechane for i in df["Volume"]]
    df["High"] = [float(i)/moneyMechane for i in df["High"]]
    df["Open"] = [float(i)/moneyMechane for i in df["Open"]]
    df["Close"] = [float(i)/moneyMechane for i in df["Close"]]

    # df.drop(df.columns[[0,3,5,6, 7,8,9]], axis=1, inplace=True)
    df.drop(df.columns[[0,3,5,6, 7,8]], axis=1, inplace=True)
    return df

def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def build_model2(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))
        model.add(Dense(1,init='uniform',activation='linear'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model


def get_df(stock_name):
    today = datetime.date.today()
    file_name = stock_name+'_stock_%s.csv' % today
    if os.path.isfile(file_name):
        df = pd.DataFrame.from_csv(file_name, header=0)
    else:
        df = get_stock_data(stock_name,0)
        df.head()
        df.to_csv(file_name)
    return df


stock_name = 'GOOG'
df = get_df(stock_name)
# df = get_stock_data(stock_name,0)
# df.head()
#
#
# today = datetime.date.today()
# file_name = stock_name+'_stock_%s.csv' % today
# df.to_csv(file_name)

df['High'] = df['High'] / 100
df['Open'] = df['Open'] / 100
df['Close'] = df['Close'] / 100
# df.head(5)


window = 5
X_train, y_train, X_test, y_test = load_data(df[::-1], window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

# model = build_model([3,window,1])
model = build_model2([3,window,1])

model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=500,
    validation_split=0.1,
    verbose=1)


trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))


print(X_test[len(X_test)-1])
diff=[]
ratio=[]
p = model.predict(X_test)

# last_prediction1 = p[len(p)-1][0]
# last_prediction2 = p[len(p)-2][0]
# last_prediction3 = p[len(p)-3][0]
# last_prediction4 = p[len(p)-4][0]
# last_prediction5 = p[len(p)-5][0]
# last_real_close = X_test[len(X_test)-1][0]
# last_real_close = last_real_close[len(last_real_close)-1]
# new_prediction = [
#     [last_real_close,0,last_prediction5],
#     [last_prediction5,0,last_prediction4],
#     [last_prediction4,0,last_prediction3],
#     [last_prediction3,0,last_prediction2],
#     [last_prediction2,0,last_prediction1],
# ]
# print(new_prediction)
# np.append(X_test, new_prediction)
# print(X_test[len(X_test)-1])
# print(p[len(p)-1])
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))
    # print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))

import matplotlib.pyplot as plt2

plt2.plot(p,color='red', label='prediction')
plt2.plot(y_test,color='blue', label='y_test')
plt2.legend(loc='upper left')
plt2.show()
