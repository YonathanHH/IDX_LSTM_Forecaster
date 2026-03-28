import pandas as pd
import plotly.graph_objs as go

from models import prepare_sequences

def make_forecast_figure(df, results, window_size, steps_ahead, title):
    prices = df["Close"].values
    idx = df.index

    X_all, _, _ = prepare_sequences(prices, window_size)
    split = int(len(X_all) * 0.8)
    test_start_idx = split + window_size

    last_ts = idx[-1]

    # ✅ Fix: compute freq as a pd.DateOffset or Timedelta safely
    if len(idx) >= 2:
        freq = idx[-1] - idx[-2]
    else:
        freq = pd.Timedelta(days=1)

    # ✅ Fix: build future_idx using pd.tseries.frequencies or timedelta
    future_idx = pd.date_range(
        start=last_ts + freq,
        periods=steps_ahead,
        freq=freq,
    )

    fig = go.Figure()

    # Candlestick: Train
    fig.add_trace(go.Candlestick(
        x=idx[:test_start_idx],
        open=df["Open"].values[:test_start_idx],
        high=df["High"].values[:test_start_idx],
        low=df["Low"].values[:test_start_idx],
        close=df["Close"].values[:test_start_idx],
        name="Train",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    # Candlestick: Test
    fig.add_trace(go.Candlestick(
        x=idx[test_start_idx:],
        open=df["Open"].values[test_start_idx:],
        high=df["High"].values[test_start_idx:],
        low=df["Low"].values[test_start_idx:],
        close=df["Close"].values[test_start_idx:],
        name="Test (actual)",
        increasing_line_color="#80cbc4",
        decreasing_line_color="#ffab91",
    ))

    # Test predicted line
    fig.add_trace(go.Scatter(
        x=idx[test_start_idx:],
        y=results["y_test_pred"],
        mode="lines",
        name="Test (predicted)",
        line=dict(color="yellow", dash="dot", width=1.5),
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_idx,
        y=results["future"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="orange", width=2),
        marker=dict(size=6),
    ))

# ✅ Use add_shape instead — fully compatible with datetime + candlestick
    fig.add_shape(
        type="line",
        x0=str(last_ts),
        x1=str(last_ts),
        y0=0,
        y1=1,
        xref="x",
        yref="paper",       # y spans full chart height (0=bottom, 1=top)
        line=dict(
            color="gray",
            dash="dash",
            width=1.5,
        ),
    )

# Add the annotation separately
    fig.add_annotation(
        x=str(last_ts),
        y=1,
        xref="x",
        yref="paper",
        text="Forecast start",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="gray", size=11),
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price (IDR)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig