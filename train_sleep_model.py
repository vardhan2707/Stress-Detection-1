"""
Train and save a sleep impact prediction model.

The model learns how daily stress patterns relate to that night's sleep:
- Features per day: stress stats + previous night's sleep.
- Targets per day: tonight's total sleep hours and deep sleep percentage.

Artifacts saved in the existing `models/` directory:
- sleep_model.pkl       (Multi-output RandomForestRegressor)
- sleep_scaler.pkl      (StandardScaler for input features)
- sleep_metadata.pkl    (feature names, metrics, etc.)
"""

import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "Final_CSVs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training dataset not found at {path}. "
            "Create `sleep_stress_dataset.csv` in Final_CSVs/ "
            "or use the sample provided with the project."
        )
    df = pd.read_csv(path)
    return df


def train_sleep_model() -> None:
    print("=" * 70)
    print("Training Sleep Impact Model (predict nightly sleep from stress)")
    print("=" * 70)

    dataset_path = os.path.join(CSV_DIR, "sleep_stress_dataset.csv")
    df = load_dataset(dataset_path)

    # Required feature and target columns
    feature_cols: List[str] = [
        "avg_stress",
        "max_stress",
        "high_stress_count",
        "hr_mean",
        "rmssd_mean",
        "scl_mean",
        "prev_total_sleep_hours",
        "prev_deep_sleep_percent",
        "prev_sleep_efficiency",
    ]
    target_cols: List[str] = [
        "next_total_sleep_hours",
        "next_deep_sleep_percent",
    ]

    missing_features = [c for c in feature_cols if c not in df.columns]
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_features or missing_targets:
        msg_parts = []
        if missing_features:
            msg_parts.append(f"missing feature columns: {', '.join(missing_features)}")
        if missing_targets:
            msg_parts.append(f"missing target columns: {', '.join(missing_targets)}")
        raise ValueError(
            "sleep_stress_dataset.csv schema mismatch — "
            + "; ".join(msg_parts)
        )

    X = df[feature_cols].values
    y = df[target_cols].values

    # Handle NaNs / infs just like in the main model
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    base_reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model = MultiOutputRegressor(base_reg)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate per target
    mae_hours = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_deep = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    r2_hours = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_deep = r2_score(y_test[:, 1], y_pred[:, 1])

    print("\nEvaluation on hold-out set:")
    print(f"  MAE (total sleep hours):   {mae_hours:.3f}")
    print(f"  MAE (deep sleep percent):  {mae_deep:.3f}")
    print(f"  R²  (total sleep hours):   {r2_hours:.3f}")
    print(f"  R²  (deep sleep percent):  {r2_deep:.3f}")

    # Persist artifacts
    joblib.dump(model, os.path.join(MODEL_DIR, "sleep_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "sleep_scaler.pkl"))

    metadata = {
        "model_type": "MultiOutputRandomForestRegressor",
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "n_samples": int(len(df)),
        "mae_total_sleep_hours": float(mae_hours),
        "mae_deep_sleep_percent": float(mae_deep),
        "r2_total_sleep_hours": float(r2_hours),
        "r2_deep_sleep_percent": float(r2_deep),
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, "sleep_metadata.pkl"))

    print("\n✓ Sleep impact model saved in `models/` as:")
    print("   - sleep_model.pkl")
    print("   - sleep_scaler.pkl")
    print("   - sleep_metadata.pkl")


if __name__ == "__main__":
    train_sleep_model()

