def make_signal(current_price, future_prices, rmse):
    """
    Simple heuristic signal + confidence, for educational use only.
    """
    current_price = float(current_price)  # ✅ ensure scalar
    rmse = float(rmse)
    final_price = float(future_prices[-1])
    ret = (final_price - current_price) / current_price

    buy_th, sell_th = 0.02, -0.02  # ±2 %
    if ret > buy_th:
        signal = "BUY"
    elif ret < sell_th:
        signal = "SELL"
    else:
        signal = "HOLD"

    mean_price = (current_price + final_price) / 2
    norm_rmse = rmse / mean_price if mean_price > 0 else 0
    confidence = max(0.0, 1.0 - norm_rmse)

    return {
        "signal": signal,
        "expected_return": ret,
        "confidence": confidence,
        "final_price": final_price,
    }