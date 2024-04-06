import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

#@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Load LSTM model
regressor = load_model('lstm_model.h5')

# Prepare data for LSTM
dataset_test = data[['Date', 'Close']].copy()
dataset_test['Date'] = pd.to_datetime(dataset_test['Date'])
dataset_test.set_index('Date', inplace=True)
sc = MinMaxScaler(feature_range=(0, 1))
testing_set_scaled = sc.fit_transform(dataset_test)

# Make predictions with LSTM for the next 100 days
X_test = []
for i in range(10, len(testing_set_scaled)):
    X_test.append(testing_set_scaled[i-10:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

latest_data = testing_set_scaled[-10:]

predictions = []
for i in range(100):
    X_test = np.array(latest_data[-10:])
    X_test = np.reshape(X_test, (1, 10, 1))
    pred_price = regressor.predict(X_test)
    pred_price = sc.inverse_transform(pred_price)
    predictions.append(pred_price[0][0])
    
    # Update latest_data with the actual data for the next prediction
    latest_data = np.vstack((latest_data, [testing_set_scaled[-100+i, 0]]))
    latest_data = latest_data[1:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=dataset_test.index, y=dataset_test['Close'], name="Actual"))
fig.add_trace(go.Scatter(x=dataset_test.index[10:], y=predicted_stock_price.flatten(), name="Predicted"))
st.plotly_chart(fig)
# Plot LSTM predictions for next 100 days
st.subheader('Predicted Prices for Next 100 Days')
st.subheader('Predicted Prices')
st.write(f"before 100 days: {predictions[-100]}")
st.write(f"After 100 days: {predictions[-1]}")

st.subheader('Predicted Prices for Next 100 Days')
fig = go.Figure()
fig.add_trace(go.Scatter(x=dataset_test.index, y=dataset_test['Close'], name="Actual"))
fig.add_trace(go.Scatter(x=pd.date_range(start=dataset_test.index[-1], periods=100, freq='D'), y=predictions, name="Predicted"))
st.plotly_chart(fig)
