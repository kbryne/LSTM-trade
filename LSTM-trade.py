import yfinance as yf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# fetching data
stock = yf.Ticker("CRAYN.OL")
data = stock.history(period="max")
#print(data.head())

scale_features = ['Open', 'High', 'Low', 'Close', 'Volume']
no_scale_features = ['Dividends', 'Stock Splits']

# scaling the correct features
sc = MinMaxScaler(feature_range=(0, 1))
scaled_data = sc.fit_transform(data[scale_features])
print(scaled_data)

# concatinate the scaled data with the unscaled data
scaled_df = pd.DataFrame(scaled_data, columns=scale_features, index=data.index)
final_data = pd.concat([scaled_df, data[no_scale_features]], axis=1)
print(final_data)

# Need to create a sequence of the data

def make_sequence(data, len_secuence=60):
    X = []
    y = []
    if len(data) > len_secuence:
        for i in range(len_secuence, len(data)):
            X.append(data.iloc[i-len_secuence:i])
            y.append(data.iloc[i]['Close'])
        return np.array(X), np.array(y)
    else:
        return np.array([]), np.array([])

# initializing lstm
model = Sequential()
model.add(layers.LSTM(units=50,
                      dropout=0.2,
                      recurrent_dropout=0.2,
                      ))