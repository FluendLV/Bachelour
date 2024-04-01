import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense
from sklearn.metrics import mean_squared_error
import math

# Load the dataset
df = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=["Date"])

# Splitting the dataset into training and test sets
training_set = df['2012-01-01':'2016-12-31'].iloc[:, 1:2].values  # Assuming column 1 is 'High'
test_set = df['2017':].iloc[:, 1:2].values

# Scaling the data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating data structures for training
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the GRU model
model = Sequential([
    GRU(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    GRU(units=80, return_sequences=True),
    Dropout(0.2),
    GRU(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Preparing the test data
inputs = df['High'].values
inputs = inputs[len(inputs) - len(test_set) - 60:]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Making predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plotting the results
plt.figure(figsize=(10,6))
plt.plot(test_set, color='red', label='Real AAPL Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted AAPL Stock Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# RMSE calculation for the predictions
rmse = math.sqrt(mean_squared_error(test_set, predicted_stock_price))
print("The root mean squared error is {}.".format(rmse))

# Assuming the rest of your script is correct

# Predicting future values, feeding the last prediction back into the model
last_60_days = training_set_scaled[-60:].reshape((1, 60, 1))
future_predictions = []

for _ in range(30):  # Predicting the next 30 days as an example
    next_day_prediction = model.predict(last_60_days)
    future_predictions.append(next_day_prediction[0, 0])
    # Reshape and append the prediction for the next iteration
    last_60_days = np.append(last_60_days[:, 1:, :], next_day_prediction.reshape(1, 1, 1), axis=1)

future_predictions = sc.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plotting the future predictions
plt.figure(figsize=(10,6))
plt.plot(range(30), future_predictions, color='blue', label='Predicted Future Stock Price')
plt.title('Future Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

