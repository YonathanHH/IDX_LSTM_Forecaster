import re
import streamlit as st

from config import WINDOW_SIZE, STEPS_AHEAD, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE
from data import to_yahoo_symbol, fetch_ohlc, get_target_series
from models import train_eval_forecast
from signals import make_signal
from plots import make_forecast_figure
from logging_utils import log_experiment_csv
st.set_page_config(page_title="IDX LSTM Forecaster", layout="wide")

st.title("LSTM-based Indonesian Stock Forecaster (Educational)")

user_input = st.text_input(
    "Input the Indonesian stocks that you want to forecast, for example BBCA, GOTO, TLKM, ADMR",
    "BBCA",
)

timeframes = st.multiselect(
    "Choose timeline to Analyze",
    ["Hourly", "Daily", "Weekly"],
    default=["Daily"],
)

epochs = st.slider("Training epochs", 10, 100, DEFAULT_EPOCHS, 10)
batch_size = st.selectbox("Batch size", [8, 16, 32], index=1)

st.caption(
    "Disclaimer: The application is only for educational purposes and is not "
    "an investment recommendation."
)

if st.button("Run forecasting"):
    idx_ticker = user_input.strip().upper()

    if not re.fullmatch(r"[A-Z]{3,5}", idx_ticker):
        st.error("Gunakan kode saham IDX yang benar, misalnya BBCA, TLKM, ADMR, GOTO.")
        st.stop()

    ya_symbol = to_yahoo_symbol(idx_ticker)
    st.write(
        f"IDX ticker: **{idx_ticker}** → Yahoo Finance symbol: **{ya_symbol}** "
        "(format <TICKER>.JK)"
    )

    for tf in timeframes:
        if tf == "Hourly":
            interval = "1h"
            tf_prefix = "Hourly"
        elif tf == "Daily":
            interval = "1d"
            tf_prefix = "Daily"
        else:
            interval = "1wk"
            tf_prefix = "Weekly"

        with st.spinner(f"Fetching {tf_prefix.lower()} data & training LSTM..."):
            df = fetch_ohlc(ya_symbol, interval=interval, candles=240)
            if len(df) < WINDOW_SIZE + STEPS_AHEAD:
                st.warning(
                    f"Data {tf_prefix.lower()} untuk {idx_ticker} tidak cukup untuk "
                    "membuat sequence (butuh minimal WINDOW_SIZE + STEPS_AHEAD)."
                )
                continue

            series = get_target_series(df, target_col="Close")
            results = train_eval_forecast(
                series,
                window_size=WINDOW_SIZE,
                epochs=epochs,
                batch_size=batch_size,
                steps_ahead=STEPS_AHEAD,
            )

            log_experiment_csv(
                ticker=idx_ticker,
                interval=interval,
                window_size=WINDOW_SIZE,
                epochs=epochs,
                batch_size=batch_size,
                mse=results["mse"],
                rmse=results["rmse"],
                steps_ahead=STEPS_AHEAD,
            )

            fig = make_forecast_figure(
                df,
                results,
                WINDOW_SIZE,
                STEPS_AHEAD,
                f"{tf_prefix} Close & LSTM Forecast - {idx_ticker}",
            )
            st.plotly_chart(fig, use_container_width=True)

            current_price = float(df["Close"].iloc[-1])
            sig = make_signal(current_price, results["future"], results["rmse"])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current price", f"{current_price:,.2f}")
            with col2:
                st.metric(
                    f"{tf_prefix} forecast (step {STEPS_AHEAD})",
                    f"{sig['final_price']:,.2f}",
                )
            with col3:
                st.metric("Expected return", f"{sig['expected_return']*100:.2f} %")
            with col4:
                st.metric("Confidence (heuristic)", f"{sig['confidence']*100:.1f} %")

            st.info(
                f"{tf_prefix} signal: **{sig['signal']}** | "
                f"MSE: {results['mse']:.2f}, RMSE: {results['rmse']:.2f}"
            )