#imports
import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

#Get the current date/time
today = date.today()

#setup our timeframe
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=5000)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

#get the dataframe from YFinance (Apple stock prices from 5000 days ago till the current date)
data = yf.download('AAPL', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)

#setup the dataframe for our project
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close",
             "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)

#Print the data
print(data.tail())

#setup visualization
figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"],
                                        high=data["High"],
                                        low=data["Low"],
                                        close=data["Close"])])
figure.update_layout(title = "Apple Stock Price Analysis",
                     xaxis_rangeslider_visible=False)

#Display CandleStick chart
figure.show()

#Now let’s have a look at the correlation of all the columns,
#with the Close column as it is the target column
correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))

#Now I will start with training an LSTM model for predicting stock prices.
#I will first split the data into training and test sets
x = data[["Open", "High", "Low", "Volume"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

#Now I will prepare a neural network architecture for LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#return model summary
model.summary()

#Train
model.compile(optimizer='adam', loss='mean_squared_error')

#model training regiment, epochs set to 5 for time conservation on non-gpu enabled machines
model.fit(xtrain, ytrain, batch_size=1, epochs=5)

#try to predict the following days closing value using this given data sample
#features = [Open, High, Low, Volume]
features = np.array([[177.089996, 180.419998, 177.070007, 74919600]])

print("Open: 177.089996,\n"
      "High: 180.419998,\n"
      "Low: 177.070007,\n"
      "Volume: 74919600")

#return the prediction
print("\nPredicted Close: ")
print(model.predict(features))