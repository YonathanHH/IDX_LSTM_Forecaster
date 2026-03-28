import csv
from datetime import datetime
from config import EXPERIMENT_CSV, EXPERIMENTS_DIR

def log_experiment_csv(
    ticker, interval, window_size, epochs, batch_size, mse, rmse, steps_ahead
):
    """
    Simple CSV experiment log for your runs.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    exists = EXPERIMENT_CSV.exists()

    with open(EXPERIMENT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "timestamp",
                "ticker",
                "interval",
                "window_size",
                "epochs",
                "batch_size",
                "steps_ahead",
                "mse",
                "rmse",
            ])
        writer.writerow([
            datetime.utcnow().isoformat(),
            ticker,
            interval,
            window_size,
            epochs,
            batch_size,
            steps_ahead,
            mse,
            rmse,
        ])