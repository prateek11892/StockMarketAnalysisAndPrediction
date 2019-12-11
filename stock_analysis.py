import yfinance as yf
from datetime import date
import pandas as pd
import numpy as np
import xgboost as xgb
import copy

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def fetchData(symbol):
	data = yf.download(symbol, date(2016,1,1), date(2018,12,31))
	data.to_csv(symbol+'.csv')

def plotGraph(df, title='Stock Prices'):
	ax = df.plot(title=title)
	ax.set_xlabel("Date")
	ax.set_ylabel("Price")
	plt.show()

def normalizedData(df):
	return df/df.ix[0,:]

def createDataFrame(stockList,start_date,end_date):	
	dates=pd.date_range(start_date,end_date)
	df = pd.DataFrame(index=dates)
	for stock in stockList:
		data = pd.read_csv(stock+'.csv', index_col="Date",parse_dates=True, usecols=['Date','Close'],na_values=['nan'])
		data = data.rename(columns={'Close':stock})

		df = df.join(data)
		df = df.dropna()

	return df

def plotRollingMean(mergedData,stock):

	ax = mergedData[stock].plot(title=stock+" rolling mean", label=stock)
	rollingMean = getRollingMean(mergedData[stock], window = 20)
	rollingMean.plot(label = "Rolling Mean",ax=ax)
	ax.set_xlabel("Date")
	ax.set_ylabel("Price")
	ax.legend(loc="upper right")
	plt.show()

def getRollingMean(values, window):
	rollingMean = values.rolling(window).mean()
	return rollingMean

def getRollingStd(values, window):
	rollingStd = values.rolling(window).std()
	return rollingStd

def getBollingerBands(rollMean, rollStd):

	upperBand = rollMean + rollStd*2
	lowerBand = rollMean - rollStd*2
	return upperBand, lowerBand

def plotBollingerBand(data, rollMean, upper_Band, lower_Band,stock):
	ax = data.plot(title = "Bollinger Bands", label = stock)
	rollMean.plot(label="RollingMean", ax=ax)
	upper_Band.plot(label="upper band", ax=ax)
	lower_Band.plot(label="lower band", ax=ax)
	ax.set_xlabel("Date")
	ax.set_ylabel("Price")
	ax.legend(loc="upper right")
	plt.show()

def computeDailyReturns(df):
	daily_returns = (df/df.shift(1))-1
	return daily_returns

def plotDailyReturns(dailyReturnsGoog, title, xlabel, label):

	ax = dailyReturnsGoog.plot(title = "Daily Returns", label = label)
	ax.set_xlabel(xlabel)
	ax.set_ylabel("Daily Returns")
	ax.legend(loc="upper right")
	plt.show()

def rsi(values):
    up = values[values>0].mean()
    down = -1*values[values<0].mean()
    return 100 * up / (up + down)

def prepareTrainingData(data, stock):
	X = data.values
	ind = list(data.columns).index(stock)
	y = []
	for i in range(X.shape[0]-1):
		if (X[i+1,ind]-X[i,ind])>0:
			y.append(1)
		else:
			y.append(0)
	y = np.array(y)
	X = X[:-1]
	return X,y

def split_train_test(X,y):
    split_ratio=0.7
    train_size = int(round(split_ratio * X.shape[0]))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = xgb.XGBClassifier()
    #model = DecisionTreeClassifier(criterion = "entropy",max_depth=3, min_samples_leaf=5) 
    model.fit(X_train, y_train)
    return model
  
def predict(model,X_test,y_test):
	y_pred = model.predict(X_test)
	y_pred = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))

#stockList = ['GOOG','XOM','AAPL']
stockList = ['AAPL']
stock = 'AAPL'
start_date='2016-01-01'
end_date='2018-12-31'

'''
for stock in stockList:
	fetchData(stock)
'''
# Getting merged data in dataframe
mergedData = createDataFrame(stockList,start_date,end_date)
df = copy.deepcopy(mergedData)
# Normalizing data to have the same starting point
normData = normalizedData(mergedData)

plotGraph(mergedData, 'Stock Price' )
plotGraph(normData, 'Normalized Data')
for stock in stockList:
	plotRollingMean(mergedData,stock)

for stock in stockList:
	df = mergedData[stock]
	
	rollMean = getRollingMean(mergedData[stock], window = 20)
	rollStd = getRollingStd(mergedData[stock], window = 20)
	lma = getRollingMean(mergedData[stock],window=50)
	mergedData['RollMean_'+stock] = rollMean
	mergedData['LMA_'+stock] = lma
	mergedData['RollStd_'+stock] = rollStd
	upper_Band, lower_Band = getBollingerBands(rollMean, rollStd)
	mergedData['UpperBand_'+stock] = upper_Band
	mergedData['LowerBand_'+stock] = lower_Band
	

# For plotting Bollinger Bands
for stock in stockList:
	plotBollingerBand(mergedData[stock], mergedData['RollMean_'+stock], mergedData['UpperBand_'+stock], mergedData['LowerBand_'+stock],stock)

# For computing Daily Returns on the stock Close price
for stock in stockList:
	dailyReturns = computeDailyReturns(mergedData[stock])
	mergedData['DailyReturns_'+stock] = dailyReturns
	plotDailyReturns(dailyReturns, title = "Daily Returns", xlabel = "Daily Returns",label = stock)

# For computing momentum and RSI value on the stock Close price
for stock in stockList:
	mergedData['Momentum_'+stock] = (mergedData[stock]-mergedData[stock].shift(1)).fillna(0)
	mergedData['RSI_'+stock] = (mergedData['Momentum_'+stock].rolling(center=False,window=14).apply(rsi).fillna(0))

mergedData = mergedData.dropna()
mergedData.to_csv('abc.csv')
print(mergedData)

X,y = prepareTrainingData(mergedData, stock)
print(y)
X_train, X_test, y_train, y_test = split_train_test(X,y)
model = train_model(X_train, y_train)
predict(model, X_test, y_test)
