import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import RootMeanSquaredError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# fetching data
stock = yf.Ticker("EQNR.OL")
#data = stock.history(period="max")
data = stock.history(start="2017-11-08", end="2023-08-01")
scale_features = ['Open', 'High', 'Low', 'Close', 'Volume']
no_scale_features = ['Dividends', 'Stock Splits']
print(data)
# scaling the correct features
sc = MinMaxScaler(feature_range=(0, 1))
scaled_data = sc.fit_transform(data[scale_features])

# concatinate the scaled data with the unscaled data
scaled_df = pd.DataFrame(scaled_data, columns=scale_features, index=data.index)
final_data = pd.concat([scaled_df, data[no_scale_features]], axis=1)

# Need to create a sequence of the data
def make_sequence(data, len_sequence=60):
    X = []
    y = []
    if len(data) >= len_sequence:
        for i in range(len_sequence, len(data)):
            X.append(data.iloc[i-len_sequence:i])
            y.append(data.iloc[i]['Close'])
        return np.array(X), np.array(y)
    else:
        return np.array([]), np.array([])

X, y = make_sequence(final_data, len_sequence=60)

X_train, X_test_temp, y_train, y_test_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test_temp, y_test_temp, test_size=0.5, shuffle=False)

# initializing lstm
model = Sequential()

model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

model.add(LSTM(units=50,
               return_sequences=True))
model.add(LSTM(units=50,
               return_sequences=True))
model.add(LSTM(units=50,
               return_sequences=False))
model.add(Dense(1))

# Compile the LSTM model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[RootMeanSquaredError()])

# Training model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

test_loss = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss)

# predict the future stock price

# fetching the last 60 days and making a numpy array
last_sequence = final_data[-60:].values.reshape(1, 60, final_data.shape[1])

future_predictions = []
num_days_to_predict = 30

for day in range(num_days_to_predict):
    # predict the next price
    next_price = model.predict(last_sequence)
    # append the predicted price to list of future predicitons
    future_predictions.append(next_price[0, 0])
    # update the last sequence to include new predictions
    last_sequence = np.roll(last_sequence, -1, axis=1)
    # first sequence, last data point in sequence, all features
    last_sequence[0, -1, :] = next_price

"""# plot the stock price
plt.figure(figsize=(12, 8))
plt.plot(data.index, data['Close'], label='Close price')
plt.xlabel('Year')
plt.ylabel('Share price')
plt.legend()
plt.show()"""

# Plot historical and predicted stock prices
plt.figure(figsize=(12, 8))
plt.plot(data.index, data['Close'], label='Historical Close Price')

# Creates a date range for future predictions
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=num_days_to_predict)

# Rescales predictions back to original values
dummy_scaled_data = np.zeros((len(future_predictions), len(scale_features)))

dummy_scaled_data[:, 3] = future_predictions

future_predictions_scaled = sc.inverse_transform(dummy_scaled_data)[:, 3]

# Append the predicted values to the plot
plt.plot(future_dates, future_predictions_scaled, label='Predicted Close Price', color='red')

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Historical and Predicted Stock Prices')
plt.legend()
plt.show()