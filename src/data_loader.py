"""Load the dataset and run basic quality checks."""
import pandas as pd
from .config import DATA_PATH, ALT_DATA_PATH, TARGET, ID_COL


def load_data(path=DATA_PATH) -> pd.DataFrame:
    if not path.exists() and ALT_DATA_PATH.exists():
        path = ALT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download 'Hotel Reservations.csv' from the Kaggle dataset 'Hotel Reservations Classification Dataset' "
            "and place it in the data/ folder."
        )
    return pd.read_csv(path)


def quality_report(df: pd.DataFrame) -> dict:
    """Return a few simple facts about the data (shape, nulls, duplicates, class balance)."""
    return {
        "shape": df.shape,
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated(subset=[c for c in df.columns if c != ID_COL]).sum()),
        "class_balance": df[TARGET].value_counts(normalize=True).round(4).to_dict(),
    }
