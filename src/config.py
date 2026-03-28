from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

WINDOW_SIZE = 60          # lookback steps
STEPS_AHEAD = 20          # forecast horizon

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 16

EXPERIMENT_CSV = EXPERIMENTS_DIR / "experiments.csv"