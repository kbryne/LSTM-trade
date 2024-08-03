import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import RootMeanSquaredError
import pandas as pd
import numpy as np

# fetching data
stock = yf.Ticker("CRAYN.OL")
data = stock.history(period="max")

scale_features = ['Open', 'High', 'Low', 'Close', 'Volume']
no_scale_features = ['Dividends', 'Stock Splits']

# scaling the correct features
sc = MinMaxScaler(feature_range=(0, 1))
scaled_data = sc.fit_transform(data[scale_features])

# concatinate the scaled data with the unscaled data
scaled_df = pd.DataFrame(scaled_data, columns=scale_features, index=data.index)
final_data = pd.concat([scaled_df, data[no_scale_features]], axis=1)


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

X, y = make_sequence(final_data, len_secuence=60)

X_train, X_test_temp, y_train, y_test_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test_temp, y_test_temp, test_size=0.5, shuffle=False)
print(y_test)

# initializing lstm
model = Sequential()

model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

model.add(LSTM(units=50,
               dropout=0.2,
               recurrent_dropout=0.2,
               return_sequences=True))
model.add(LSTM(units=50,
               dropout=0.2,
               recurrent_dropout=0.2,
               return_sequences=True))
model.add(LSTM(units=50,
               return_sequences=False))
model.add(Dense(1))

# Compile the LSTM model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[RootMeanSquaredError()])

