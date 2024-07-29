import yfinance as yf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# fetching data
stock = yf.Ticker("CRAYN.OL")
data = stock.history(period="max")
print(data)

# scaling the closing prices
sc = MinMaxScaler(feature_range=(0, 1))
scaled_data = sc.fit_transform(data['Close'].values.reshape(-1, 1))
print(scaled_data)

# initializing lstm
model = Sequential()
model.add(layers.LSTM(units=50,
                      dropout=0.2,
                      recurrent_dropout=0.2,
                      ))