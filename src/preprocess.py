"""Cleaning, encoding, splitting and scaling.
KNN is distance based, so scaling is mandatory (Day 4/5: Euclidean distance)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import TARGET, ID_COL, CATEGORICAL_COLS, RANDOM_STATE, TEST_SIZE


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[ID_COL], errors="ignore")        # an ID carries no signal
    df = df.drop_duplicates()
    df[TARGET] = df[TARGET].map({"Not_Canceled": 0, "Canceled": 1})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)


def split_and_scale(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)                 # fit on TRAIN only -> no data leakage
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler, list(X.columns)


def prepare(raw_df: pd.DataFrame):
    df = encode(engineer_features(clean(raw_df)))
    return split_and_scale(df)
