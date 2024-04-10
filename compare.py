import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load historical stock price data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Preprocess data for training
def preprocess_data(data):
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data

# Create features and target variable
def create_features_target(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data['Returns'].values[i:i+window_size])
        y.append(data['Close'].values[i+window_size])
    X, y = np.array(X), np.array(y)
    return X, y

# Split data into training and testing sets
def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test

# Train Linear Regression model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Train Random Forest model
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return model

# Train LSTM model
def train_lstm(X_train, y_train, window_size):
    model = Sequential()
    model.add(LSTM(50, input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=100, batch_size=32)
    return model

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse, predictions

# Plot actual vs. predicted prices
def plot_results(actual, predicted, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

# Main function
def main():
    st.title('Stock Price Prediction')

    # Sidebar inputs
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", value='AAPL')
    start_date = st.sidebar.text_input("Enter Start Date (YYYY-MM-DD)", value='2010-01-01')
    end_date = st.sidebar.text_input("Enter End Date (YYYY-MM-DD)", value='2022-01-01')
    window_size = st.sidebar.number_input("Enter Window Size", value=30)
    test_size = st.sidebar.number_input("Enter Test Size", value=0.2)

    # Load and preprocess data
    data = load_data(ticker, start_date, end_date)
    data = preprocess_data(data)

    # Create features and target variable
    X, y = create_features_target(data, window_size)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    # Train models
    st.subheader('Model Training')
    linear_regression_model = train_linear_regression(X_train, y_train)
    random_forest_model = train_random_forest(X_train, y_train)
    lstm_model = train_lstm(X_train, y_train, window_size)

    # Evaluate models
    st.subheader('Model Evaluation')
    mse_lr, predictions_lr = evaluate_model(linear_regression_model, X_test, y_test)
    mse_rf, predictions_rf = evaluate_model(random_forest_model, X_test, y_test)
    mse_lstm, predictions_lstm = evaluate_model(lstm_model, X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test)

    # Display results
    st.write("Mean Squared Error (MSE)")
    st.write("Linear Regression:", mse_lr)
    st.write("Random Forest:", mse_rf)

    st.subheader('Predictions vs. Actual Prices')
    plot_results(y_test, predictions_lr, 'Linear Regression')
    plot_results(y_test, predictions_rf, 'Random Forest')

if __name__ == "__main__":
    main()
