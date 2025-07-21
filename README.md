
# 📈 Live Stock Market Prediction GUI (Tkinter + LSTM)

A real-time stock market prediction desktop app built with **Python**, using **LSTM (Long Short-Term Memory)** neural networks, **YahooQuery** for live data fetching, and **Tkinter** for the GUI interface. The project predicts future stock prices based on real-time data and visualizes the results graphically.

---

## 🚀 Features

- 🔥 Live stock price updates (every 60 seconds)
- 📊 Real-time graph plotting (Actual vs Predicted)
- ⚡ LSTM-based price prediction
- 🖥️ Interactive desktop GUI using Tkinter
- 📥 Fetches historical and live data via YahooQuery

---

## 🛠️ Technologies Used

- Python
- Tkinter (GUI)
- YahooQuery (live stock data)
- Pandas & NumPy (data manipulation)
- Scikit-learn (data preprocessing)
- TensorFlow / Keras (LSTM neural network)
- Matplotlib (plotting graphs)
- Multithreading (background live updates)

---

## 📦 Installation

1. **Clone this repository:**

```bash
git clone https://github.com/your-username/stock-market-prediction-gui.git
cd stock-market-prediction-gui
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install yahooquery pandas numpy scikit-learn tensorflow matplotlib
```

3. **Run the app:**

```bash
python main.py
```

---

## 🎬 How It Works

- Enter a **stock ticker** (e.g. `MRF.NS` for MRF stock on NSE India).
- Click **Start Prediction**.
- App fetches **1 year of historical data**, trains an **LSTM model**, and starts fetching **live stock data every 60 seconds**.
- The graph updates dynamically with **Actual Prices** (blue) and **Predicted Prices** (orange).

---


## 📊 Project Structure

```plaintext
project/
├── main.py            # Main application code
├── requirements.txt    # List of dependencies
├── README.md           # Project documentation
```

---

## ⚠️ Disclaimer

This application is intended **for educational and demonstration purposes only**. Do not use this tool for actual trading or financial decisions.

---

## ✨ Author

**Parth Omar**

- 📧 Email: parthomar2002@gmail.com  
- 🔗 [LinkedIn](https://www.linkedin.com/in/parth-omar-7ba086236)

---

## 📑 License

This project is open-source under the MIT License.
