"""Central settings so every script uses the same paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Kaggle's file is named "Hotel Reservations.csv" (with a space); the underscore version also works.
DATA_PATH = ROOT / "data" / "Hotel Reservations.csv"
ALT_DATA_PATH = ROOT / "data" / "Hotel_Reservations.csv"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
RESULTS_PATH = ROOT / "reports" / "results.txt"

TARGET = "booking_status"
ID_COL = "Booking_ID"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

CATEGORICAL_COLS = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]
