import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import math
from sklearn.metrics import mean_squared_error


class CustomGRUCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomGRUCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        self.Wr = self.add_weight(shape=(input_shape[-1] + self.units, self.units),
                                  initializer='glorot_uniform',
                                  name='Wr')
        self.Wz = self.add_weight(shape=(input_shape[-1] + self.units, self.units),
                                  initializer='glorot_uniform',
                                  name='Wz')
        self.W = self.add_weight(shape=(input_shape[-1] + self.units, self.units),
                                 initializer='glorot_uniform',
                                 name='W')

        self.br = self.add_weight(shape=(self.units,), initializer='zeros', name='br')
        self.bz = self.add_weight(shape=(self.units,), initializer='zeros', name='bz')
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', name='b')

    def call(self, inputs, states):
        h_prev = states[0]

        x_h = tf.concat([inputs, h_prev], axis=-1)

        r = tf.sigmoid(tf.matmul(x_h, self.Wr) + self.br)
        z = tf.sigmoid(tf.matmul(x_h, self.Wz) + self.bz)
        h_tilde = tf.tanh(tf.matmul(tf.concat([inputs, r * h_prev], axis=-1), self.W) + self.b)

        h = (1 - z) * h_prev + z * h_tilde

        return h, [h]

class CustomGRUModel(tf.keras.Model):
    def __init__(self, units, **kwargs):
        super(CustomGRUModel, self).__init__(**kwargs)
        self.gru_cell = CustomGRUCell(units)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # Assuming inputs is shaped [batch, time, features]
        outputs = tf.keras.backend.rnn(self.gru_cell, inputs, initial_states=[tf.zeros((tf.shape(inputs)[0], self.gru_cell.units))])
        return self.dense(outputs[0])

# Load the dataset
df = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=["Date"])

# Data preprocessing
training_set = df['2012-01-01':'2016-12-31'].iloc[:, 1:2].values
test_set = df['2017':].iloc[:, 1:2].values

sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Use our custom GRU model
model = CustomGRUModel(100)  # Adjust units as needed
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Preparing test data
dataset_total = pd.concat((df["High"][:'2016'], df["High"]['2017':]), axis=0)
print(dataset_total)
inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
print(inputs)
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predicting
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(test_set[:120], color='red', label='Real AAPL Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted AAPL Stock Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AAPL Stock Price')
plt.legend()
plt.show()

# Evaluation
rmse = math.sqrt(mean_squared_error(test_set, predicted_stock_price))
print("The root mean squared error is {}.".format(rmse))

r2 = r2_score(test_set, predicted_stock_price)
print("The R-squared value is {}.".format(r2))




