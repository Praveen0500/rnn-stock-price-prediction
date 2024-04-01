# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

![image](https://github.com/Praveen0500/rnn-stock-price-prediction/assets/120218611/0fef148e-9668-4c77-ba15-d38472a83cf1)


## DESIGN STEPS

### STEP 1:Read and preprocess training data, including scaling and sequence creation.

### STEP 2: Initialize a Sequential model and add SimpleRNN and Dense layers.

### STEP 3: Compile the model with Adam optimizer and mean squared error loss.

### STEP 4:Train the model on the prepared training data.

### STEP 5:Preprocess test data, predict using the trained model, and visualize the results



## PROGRAM
```py
 Name: PRAVEEN S
 Register Number: 212222240078

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
dataset_train = pd.read_csv('trainset.csv')
dataset_train.columns
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(train_set)
training_set_scaled
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
X_train1.shape
from tensorflow.keras.layers import SimpleRNN,Dense
model = Sequential([
    SimpleRNN(50,input_shape=(60,1)),
    Dense(1)
])
model.compile(optimizer='adam',loss='mse')
model.summary()
model.fit(X_train1,y_train,epochs=100, batch_size=32)
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


```

## OUTPUT

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/Praveen0500/rnn-stock-price-prediction/assets/120218611/c19fb20e-c81f-4e84-92c8-c27ce6330e8b)

### Mean Square Error

![image](https://github.com/Praveen0500/rnn-stock-price-prediction/assets/120218611/ec5c499e-8ea3-4530-834a-e1dc7dd90202) ![image](https://github.com/Praveen0500/rnn-stock-price-prediction/assets/120218611/f96f6656-12e2-4ac9-a85c-6a8d0f539782)


## RESULT

Thus a Recurrent Neural Network model for stock price prediction program is executed successfully.
