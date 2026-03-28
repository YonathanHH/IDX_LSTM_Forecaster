# 📈 IDX LSTM Forecaster

A deep learning-powered stock forecasting app for **Indonesian Stock Exchange (IDX)** tickers, built with LSTM (Keras) and deployed via Streamlit.

> ⚠️ **Disclaimer**: This app is for **educational and experimental purposes only**. It is **not** financial or investment advice. LSTM-based forecasts on financial time series are highly uncertain.

---

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://idxlstmforecaster-byhary.streamlit.app/)

---

## Features

- 🔎 **IDX Ticker Input** — Enter any valid IDX stock code (e.g. `BBCA`, `TLKM`, `GOTO`, `ADMR`)
- 📊 **Multi-timeframe Analysis** — Hourly, Daily, and Weekly forecasts (240 candles each)
- 🕯️ **Candlestick Charts** — Interactive Plotly candlestick with Train / Test / Forecast overlays
- 🤖 **LSTM Model** — Keras LSTM trained on 80% of data, evaluated on 20% (MSE & RMSE)
- 🔮 **20-step Forecast** — Iterative multi-step price projection per timeframe
- 📉 **Buy / Sell / Hold Signal** — Heuristic signal based on expected return
- 📈 **Confidence Level** — Normalized RMSE-based heuristic confidence score
- 📝 **Experiment Logging** — All runs logged to a local CSV (`data/experiments/experiments.csv`)

---

## 🗂️ Project Structure

```
idx-lstm-forecast/
├── data/
│ └── experiments/ # Auto-created; stores experiment logs (CSV)
├── src/
│ ├── config.py # Global constants and paths
│ ├── data.py # yfinance data fetching & preprocessing
│ ├── models.py # LSTM build, train, evaluate, forecast
│ ├── plots.py # Plotly candlestick + forecast charts
│ ├── signals.py # Buy/Sell/Hold signal & confidence logic
│ ├── logging_utils.py # CSV experiment logger
│ └── app.py # Streamlit entrypoint
├── requirements.txt
└── README.md
```


---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/idx-lstm-forecast.git
cd idx-lstm-forecast
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
cd src
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Dependencies
```
streamlit>=1.30
yfinance>=0.2.40
pandas>=2.0
numpy>=1.25
plotly>=5.20
scikit-learn>=1.4
tensorflow>=2.15
```


---

## 🧠 How It Works

### 1. Data Fetching
The app uses [yfinance](https://github.com/ranaroussi/yfinance) to download the last **240 candles** for the chosen timeframe. IDX tickers are automatically suffixed with `.JK` (e.g. `BBCA` → `BBCA.JK`).

| Timeframe | Interval | Period Fetched |
|-----------|----------|----------------|
| Hourly    | `1h`     | Last 30 days   |
| Daily     | `1d`     | Last 2 years   |
| Weekly    | `1wk`    | Last 10 years  |

### 2. LSTM Model
- **Sequence window**: 60 timesteps lookback
- **Architecture**: 2× LSTM(64) layers with Dropout(0.2) + Dense(1) output
- **Train/Test split**: 80% train / 20% test
- **Evaluation**: MSE and RMSE computed on the test set
- **Production forecast**: Model is refit on full data before generating the 20-step forecast

### 3. Forecast & Signal

The model iteratively forecasts the next **20 time steps**. A heuristic signal is derived from expected return:

| Expected Return | Signal |
|-----------------|--------|
| > +2%           | 🟢 BUY  |
| < -2%           | 🔴 SELL |
| Between ±2%     | 🟡 HOLD |

Confidence is computed as:
```
confidence = max(0, 1 - (RMSE / mean_price))
```


### 4. Visualization
Each timeframe produces an interactive Plotly candlestick chart showing:
- 🕯️ **Train data** — green/red candlesticks
- 🕯️ **Test actual** — lighter candlesticks
- 📍 **Test predicted** — dotted yellow line overlay
- 🔶 **20-step forecast** — orange line + markers
- ⚡ **Forecast separator** — vertical dashed line at forecast start

---

## 📊 Example IDX Tickers

| Ticker | Company |
|--------|---------|
| `BBCA` | Bank Central Asia |
| `TLKM` | Telekomunikasi Indonesia |
| `BMRI` | Bank Mandiri |
| `BBRI` | Bank Rakyat Indonesia |
| `GOTO` | GoTo Gojek Tokopedia |
| `ADMR` | Alamtri Minerals Indonesia |
| `PGAS` | Perusahaan Gas Negara |
| `ANTM` | Aneka Tambang |

---

## 🗺️ Roadmap

- [ ] GARCH volatility bands overlaid on LSTM forecast
- [ ] Efficient Frontier multi-stock portfolio optimizer
- [ ] FastAPI backend for REST API access
- [ ] Model persistence (save/load trained `.keras` model)
- [ ] MLflow experiment tracking integration
- [ ] Docker deployment support

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@YonathanHH](https://github.com/yonathanhh)
- More Portfolio: [Github IO](https://yonathanhh.github.io/)
- LinkedIn: [Yonathan Hary](https://linkedin.com/in/yonathanhary)

---

## 🙏 Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance) for Indonesian market data
- [TensorFlow / Keras](https://keras.io) for LSTM implementation
- [Streamlit](https://streamlit.io) for rapid app deployment
- [Plotly](https://plotly.com) for interactive candlestick charts
- [scikit-learn](https://scikit-learn.org) for preprocessing and metrics
