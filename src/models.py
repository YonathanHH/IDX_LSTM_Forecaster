import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def prepare_sequences(series: np.ndarray, window_size: int = 60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.expand_dims(X, axis=-1)  # (samples, timesteps, 1)
    return X, y, scaler

def build_lstm(window_size: int) -> Sequential:
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def train_eval_forecast(
    series,
    window_size=60,
    epochs=30,
    batch_size=16,
    steps_ahead=20,
):
    """
    Train LSTM with 80/20 split, evaluate on test,
    then refit on full data and forecast next steps_ahead points.
    """
    X, y, scaler = prepare_sequences(series, window_size)
    if len(X) < 10:
        raise ValueError("Not enough data after sequence building.")

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm(window_size)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    y_pred_test_scaled = model.predict(X_test, verbose=0)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_test_inv = scaler.inverse_transform(y_pred_test_scaled).ravel()

    mse = mean_squared_error(y_test_inv, y_pred_test_inv)
    rmse = np.sqrt(mse)

    # Refit on full data for forecasting
    model_full = build_lstm(window_size)
    model_full.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # Iterative multi-step forecast
    last_window = X[-1].copy()
    future_scaled = []
    for _ in range(steps_ahead):
        pred_scaled = model_full.predict(
            last_window.reshape(1, window_size, 1),
            verbose=0,
        )[0, 0]
        future_scaled.append(pred_scaled)
        last_window = np.vstack([last_window[1:], [[pred_scaled]]])

    future_scaled = np.array(future_scaled).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_scaled).ravel()

    return {
        "history": history,
        "mse": mse,
        "rmse": rmse,
        "y_test": y_test_inv,
        "y_test_pred": y_pred_test_inv,
        "future": future_prices,
    }