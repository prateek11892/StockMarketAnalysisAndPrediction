import yfinance as yf
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from StationarityTests import StationarityTests

from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA

def fetchData(symbol):
	data = yf.download(symbol, date(2016,1,1), date(2018,12,31))
	data.to_csv(symbol+'.csv')

def createDataFrame(stockList,start_date,end_date):	
	dates=pd.date_range(start_date,end_date)
	df = pd.DataFrame(index=dates)
	for stock in stockList:
		data = pd.read_csv(stock+'.csv', index_col="Date",parse_dates=True, usecols=['Date','Close'],na_values=['nan'])
		data = data.rename(columns={'Close':stock})

		df = df.join(data)
		df = df.dropna()

	return df

def computeDailyReturns(df):
	daily_returns = (df/df.shift(1))-1
	return daily_returns

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

stockList = ['GOOG']
stock = 'GOOG'
'''
for stock in stockList:
	fetchData(stock)
'''
start_date='2016-01-01'
end_date_arima='2018-11-30'
end_date = '2018-12-31'

predictionDays = 18
mergedData = createDataFrame(stockList,start_date,end_date_arima)
mergedDataTotal = createDataFrame(stockList,start_date,end_date)
#print(mergedData['XOM'].tolist())

dailyReturns = computeDailyReturns(mergedDataTotal[stock])
#print("type:",type(dailyReturns))
actualReturns = dailyReturns.tail(predictionDays).to_numpy()
print(actualReturns)

series = mergedData[stock].tolist()

sTest = StationarityTests()
sTest.ADF_Stationarity_Test(series, printResults = True)
print("Is the time series stationary? {0}".format(sTest.isStationary))

differencedData = mergedData[stock].diff()
differencedData = differencedData.fillna(method='bfill')

'''
if sTest.isStationary == False:

	sTest = StationarityTests()
	sTest.ADF_Stationarity_Test(differencedData.tolist(), printResults = True)
	print("Is the time series stationary? {0}".format(sTest.isStationary))

autocorrelation_plot(mergedData['XOM'])
plt.show()

plot_pacf(mergedData['XOM'],lags=700)
plt.show()

'''
model = ARIMA(differencedData,order=(10,1,4))
model_fit = model.fit(disp=0)
print(model_fit.summary())

residuals = pd. DataFrame(model_fit.resid)
residuals.plot()
plt.show()


residuals.plot(kind='kde')
plt.show()

print(residuals.describe())

forecast = model_fit.forecast(steps=predictionDays)[0]
print("Forecast: %f", forecast)


print(mergedData)

differencedData = differencedData.append(pd.DataFrame(forecast))

history = [x for x in mergedData[stock]]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, 1)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1


actualReturns = actualReturns.flatten()
forecast = forecast.flatten()
count = 0;
for x,y in zip(actualReturns,forecast):
	if x > 0 and y >0 :
		count+=1
	elif x < 0 and y < 0:
		count += 1
print("Accuracy:", count/predictionDays)
