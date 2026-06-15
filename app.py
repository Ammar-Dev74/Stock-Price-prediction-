import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from sklearn.svm import SVR
# Load the trained model
lstm_model = load_model(r"C:\Users\Ammar\Desktop\stock price prediction\Stock_Price_Predicitions_model.keras")  # Make sure to load the correct model file type

st.header('Stock Price Prediction')

# Stock symbol input
stock = st.text_input('Enter stock symbol', 'GOOG')

# Load the dataset
data = pd.read_csv(r"C:\Users\Ammar\Desktop\stock price prediction\data.csv")  # Use a raw string for file path

# Show the dataset
st.subheader('Stock Data')
st.write(data)

# Prepare the data for training and testing
data_train = data['Close'][0:int(len(data) * 0.80)]
data_test = data['Close'][int(len(data) * 0.80):]

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train.values.reshape(-1,1))

pas_100_days = data_train.tail(100)
data_test_combined= pd.concat([pas_100_days,data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_combined.values.reshape(-1,1))

st.subheader('100-Day Moving Average')
ma_100_days = data['Close'].rolling(100).mean()
fig1 = plt.figure(figsize=(6,4))
plt.plot(ma_100_days, 'red', label='100-Day Moving Average')
plt.plot(data['Close'], 'green', label='Closing Price')

plt.xlabel('Data')
plt.ylabel('Price in USD ($)')
plt.legend(loc='best')
  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to avoid label cutoffs
plt.show()
st.pyplot(fig1)

st.subheader('200-Day Moving Average')
ma_200_days = data['Close'].rolling(200).mean()          
fig2=plt.figure(figsize=(6,4))
plt.plot(ma_100_days,'red',label='100-Day Moving Average')
plt.plot(ma_200_days,'blue',label='200-Day Moving Average')
plt.plot(data['Close'],'green',label='closing Price')
plt.legend(loc='best')
plt.tight_layout()  
plt.show()
st.pyplot(fig2)

# Prepare the training dataset with time steps of 100
x_test = []
y_test = []
for i in range(100,(data_test_scaled.shape[0])):
    x_test.append(data_test_scaled[i-100:i , 0])
    y_test.append(data_test_scaled[i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape for LSTM input

lstm_predictions = lstm_model.predict(x_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)
lstm_predictions = lstm_predictions.reshape(-1, 1)

svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
x_train_svm = np.arange(0,len(data_train)).reshape(-1, 1)
y_train_svm = data_train.values

# Fit the SVM model
svm_model.fit(x_train_svm, y_train_svm)

# Test data for predictions
x_test_svm = np.arange(len(data_train), len(data_train) + len(data_test)).reshape(-1, 1)
svm_predictions = svm_model.predict(x_test_svm)

st.subheader(' Actual vs Predicted values for LSTM, SVM')
fig3=plt.figure(figsize=(6, 4))
plt.plot(data_test.index, data_test.values, color='blue', label='Actual GOOG Price')
plt.plot(data_test.index, lstm_predictions, color='red', label='LSTM Predicted GOOG Price')
plt.plot(data_test.index, svm_predictions, color='green', label='SVM Predicted GOOG Price')
plt.title('GOOG Stock Price Prediction Comparison (LSTM, SVM)')
plt.xlabel('Data')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()
st.pyplot(fig3)

actual_prices = data_test.values[-min(len(lstm_predictions),len(svm_predictions)):]
actual_prices = actual_prices.flatten()
lstm_predictions = lstm_predictions.flatten()  # Flatten LSTM predictions if needed
svm_predictions = svm_predictions.flatten()  # Flatten SVM predictions

# comparison DataFrame 
comparison_df = pd.DataFrame({
    'Data': data_test.index[-len(actual_prices):],  # Slice dates to match the test data
    'Actual Price': actual_prices,
    'LSTM Predicted': lstm_predictions,
    'SVM Predicted': svm_predictions,
})
print(comparison_df.head())

st.title('Comparison of Actual Prices vs Model Predictions')


st.write(comparison_df)

st.line_chart(comparison_df.set_index('Data'))