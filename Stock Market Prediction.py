import tkinter as tk
from yahooquery import Ticker
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# Fetch historical data using yahooquery
def get_stock_data(ticker_symbol):
    t = Ticker(ticker_symbol)
    data = t.history(period='1y', interval='1d')
    if data.empty:
        return None
    df = data.reset_index()
    df = df[['date', 'close']].dropna()
    df.set_index('date', inplace=True)
    return df

# Preprocess data for LSTM
def preprocess_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)

    time_step = 60
    X_train, y_train = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X_train.append(scaled_data[i:(i + time_step), 0])
        y_train.append(scaled_data[i + time_step, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    return X_train, y_train, scaler

# Build LSTM model
def build_lstm_model(X_train, y_train):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(units=50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10)
    return model

# Predict next price
def predict_live_price(model, scaler, historical_data, live_close):
    last_60_days = historical_data[-60:].values.flatten()
    last_60_days = np.append(last_60_days, live_close)
    last_60_days = last_60_days[-60:]  # Ensure 60 data points
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    X_test = np.array([last_60_days_scaled])
    X_test = X_test.reshape(1, 60, 1)
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]

# Main GUI App
class StockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Stock Market Prediction (YahooQuery)")
        self.root.geometry("800x600")

        self.ticker_label = tk.Label(root, text="Stock Ticker (Example: MRF.NS):")
        self.ticker_label.pack(pady=10)

        self.ticker_entry = tk.Entry(root)
        self.ticker_entry.insert(0, "MRF.NS")
        self.ticker_entry.pack(pady=5)

        self.predict_button = tk.Button(root, text="Start Prediction", command=self.start_prediction)
        self.predict_button.pack(pady=10)

        self.status_label = tk.Label(root, text="", fg="green")
        self.status_label.pack(pady=5)

        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Stock Price Prediction")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack()

        self.prediction_thread = None
        self.running = False

    def start_prediction(self):
        ticker = self.ticker_entry.get()
        self.status_label.config(text="Loading data...")

        def run_prediction():
            stock_data = get_stock_data(ticker)
            if stock_data is None or stock_data.empty:
                self.status_label.config(text="Error: Invalid ticker or no data available.", fg="red")
                return

            X_train, y_train, scaler = preprocess_data(stock_data[['close']])
            model = build_lstm_model(X_train, y_train)
            historical_data = stock_data[['close']]

            self.status_label.config(text="Prediction started!", fg="green")
            self.running = True
            live_prices = list(historical_data['close'].values)
            predictions = []

            t = Ticker(ticker)

            while self.running:
                try:
                    live_data = t.history(period='1d', interval='5m')
                    if live_data.empty:
                        self.status_label.config(text="Live data unavailable. Retrying...", fg="red")
                        time.sleep(60)
                        continue

                    df = live_data.reset_index()
                    latest_price = df['close'].dropna().values[-1]

                    predicted_price = predict_live_price(model, scaler, historical_data, latest_price)

                    live_prices.append(latest_price)
                    predictions.append(predicted_price)

                    self.ax.clear()
                    self.ax.plot(live_prices, label='Actual Price', color='blue')
                    self.ax.plot(range(len(live_prices) - len(predictions), len(live_prices)),
                                 predictions, label='Predicted Price', color='orange')
                    self.ax.legend(loc='upper left')
                    self.ax.set_title("Stock Price Prediction")
                    self.ax.set_xlabel("Time")
                    self.ax.set_ylabel("Price")
                    self.canvas.draw()

                    time.sleep(60)  # Update every 60 seconds

                except Exception as e:
                    self.status_label.config(text=f"Error: {e}", fg="red")
                    self.running = False
                    break

        self.prediction_thread = threading.Thread(target=run_prediction)
        self.prediction_thread.start()

    def stop_prediction(self):
        self.running = False
        if self.prediction_thread:
            self.prediction_thread.join()

    def __del__(self):
        self.stop_prediction()

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = StockApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_prediction)
    root.mainloop()
