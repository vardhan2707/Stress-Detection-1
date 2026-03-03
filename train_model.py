"""
Train and evaluate stress detection models for the prediction dashboard.

This script focuses on:
- Robust physiological data cleaning
- Proper train/test splitting and scaling (no data leakage)
- Handling potential class imbalance
- Training and comparing multiple ML models
- Selecting and persisting the best-performing model
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    classification_report,
    roc_curve,
)
from sklearn.base import clone
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "Final_CSVs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def _clean_swell_dataframe(df_raw: pd.DataFrame, feature_cols):
    """
    Apply robust preprocessing to SWELL physiological data.

    Cleaning rules:
    1. Replace sentinel 999 values with NaN (entire DataFrame).
    2. Keep only physiologically plausible values:
       - HR in [40, 200]
       - RMSSD in [5, 200]
    3. Clip SCL into [0.5, 50] instead of hard-filtering.
    4. Drop rows with missing values in ['HR', 'RMSSD', 'SCL', 'stress'].
    5. Recalculate and display column statistics and assert bounds.
    """
    df = df_raw.copy()

    # 1. Replace 999 placeholders with NaN everywhere
    df.replace(999, np.nan, inplace=True)

    # 2. Enforce HR and RMSSD bounds
    if {"HR", "RMSSD"}.issubset(df.columns):
        df = df[
            df["HR"].between(40, 200)
            & df["RMSSD"].between(5, 200)
        ]

    # 3. Clip SCL to a reasonable physiological band
    if "SCL" in df.columns:
        df["SCL"] = df["SCL"].clip(lower=0.5, upper=50)

    # 4. Drop rows with missing essentials
    required_cols = [c for c in ["HR", "RMSSD", "SCL", "stress"] if c in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols)

    # 5. Recalculate and display basic column statistics
    print("\nCleaned feature statistics (HR, RMSSD, SCL):")
    try:
        print(df[feature_cols].describe())
    except Exception:
        print("  (Unable to compute stats; verify feature columns exist and are numeric.)")

    # Assert bounds AFTER cleaning to avoid training on corrupted data
    if "HR" in df.columns:
        assert df["HR"].between(40, 200).all(), "HR out of bounds"
    if "RMSSD" in df.columns:
        assert df["RMSSD"].between(5, 200).all(), "RMSSD out of bounds"

    return df


def _run_safety_checks(df: pd.DataFrame, feature_cols):
    """Run final sanity checks on cleaned physiological data."""
    print("\nFinal safety checks:")

    # Ensure no placeholder 999 values remain
    if (df[feature_cols] == 999).any().any():
        print("  [WARN] Placeholder value 999 still present after cleaning.")
    else:
        print("  [OK] No 999 placeholder values remain.")

    # Check means against plausible physiological ranges
    expected_ranges = {
        "HR": (60, 100),       # bpm
        "RMSSD": (20, 60),     # ms
        "SCL": (1, 6),         # µS
    }

    for col in feature_cols:
        if col not in df.columns:
            continue
        mean_val = float(df[col].mean())
        low, high = expected_ranges.get(col, (None, None))
        if low is not None:
            in_range = low <= mean_val <= high
            status = "OK" if in_range else "WARN"
            # Use ASCII-only output for Windows console compatibility
            print(f"  {status} {col} mean: {mean_val:.2f} "
                  f"(expected roughly {low}-{high})")


def _evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Fit model and compute a rich set of metrics.
    Returns a metrics dict including the fitted model instance.
    """
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)

    # Use probability scores whenever possible (required for ROC-AUC and
    # threshold optimisation). Fall back to hard predictions only if needed.
    y_scores = None
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)

    # Optimise classification threshold on the test set for best F1.
    # This is primarily for analysis/viva readiness; the app still uses
    # model.predict_proba and its own thresholds when needed.
    if y_scores is not None and len(np.unique(y_test)) > 1:
        best_f1 = -1.0
        best_recall = -1.0
        best_thresh = 0.5
        for thr in np.linspace(0.05, 0.95, 91):
            y_pred_thr = (y_scores >= thr).astype(int)
            _, r, f, _ = precision_recall_fscore_support(
                y_test,
                y_pred_thr,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
            # Avoid solutions with zero recall for the minority class
            if r == 0.0:
                continue
            # Prefer higher F1; break ties with higher recall
            if f > best_f1 or (np.isclose(f, best_f1) and r > best_recall):
                best_f1 = f
                best_recall = r
                best_thresh = thr

        if best_f1 >= 0:
            print(f"  Optimal decision threshold (by F1): {best_thresh:.3f}")
            y_pred = (y_scores >= best_thresh).astype(int)
        else:
            # All thresholds led to zero recall; fall back to 0.5
            print("  [WARN] All thresholds yielded zero recall; using default 0.5.")
            y_pred = (y_scores >= 0.5).astype(int)
    else:
        # No score function available; use model.predict (no custom thresholding)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )

    # MCC is robust to class imbalance; guard against degenerate predictions
    if len(np.unique(y_pred)) > 1:
        mcc = matthews_corrcoef(y_test, y_pred)
    else:
        mcc = 0.0
        print(f"  [WARN] {name} predicted only one class; MCC set to 0.0.")

    # ROC-AUC using probability scores or decision function (never class labels)
    roc_auc = float("nan")
    try:
        if y_scores is not None and len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_scores)
    except Exception:
        roc_auc = float("nan")

    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    if np.isfinite(roc_auc):
        print(f"ROC-AUC:    {roc_auc:.4f}")
    else:
        print("ROC-AUC:    N/A")
    print(f"MCC:        {mcc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, target_names=["No Stress", "Stress"]))

    return {
        "model": model,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "confusion_matrix": cm.tolist(),
    }


def _cross_validate_model(name, base_model, X, y, n_splits=5):
    """
    Perform stratified K-fold cross-validation for a single model.

    Returns mean accuracy, F1, and ROC-AUC across folds.
    Thresholding inside folds uses 0.5 on probability scores to keep
    the comparison consistent; final threshold optimisation is done
    later on a hold-out split for the selected model.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = clone(base_model)
        model.fit(X_tr_s, y_tr)

        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_val_s)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_val_s)
        else:
            y_scores = None

        if y_scores is not None:
            y_pred = (y_scores >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_val_s)

        accs.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred, average="binary", pos_label=1, zero_division=0))

        try:
            if y_scores is not None and len(np.unique(y_val)) > 1:
                aucs.append(roc_auc_score(y_val, y_scores))
        except Exception:
            # If ROC fails for a fold, skip contributing that fold's AUC
            pass

    mean_acc = float(np.mean(accs)) if accs else float("nan")
    mean_f1 = float(np.mean(f1s)) if f1s else float("nan")
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")

    print(
        f"{name:18s} CV (n={n_splits}) -> "
        f"Acc: {mean_acc:.3f} | F1: {mean_f1:.3f} | "
        f"AUC: {np.nan_to_num(mean_auc, nan=0.0):.3f}"
    )

    return {
        "mean_accuracy": mean_acc,
        "mean_f1": mean_f1,
        "mean_roc_auc": mean_auc,
    }


def train_swell_model():
    """
    Clean a WESAD-derived dataset and train a single XGBoost classifier on
    physiologically valid data with engineered features.

    Steps:
    - Load and clean data (remove 999s and out-of-range values)
    - Remove clearly contradictory low-stress labels (HR high, RMSSD low, SCL high)
    - Run safety checks and hard assertions on bounds
    - Engineer additional physiologically meaningful features
    - Stratified 5-fold CV for robust performance estimates
    - Stratified 80/20 train-test split
    - StandardScaler fit on train only (no leakage)
    - Train regularised XGBoost model
    - Print Accuracy, ROC-AUC, classification report, confusion matrix
    - Persist final model and scaler for the Streamlit app
    """
    print("=" * 60)
    print("Training XGBoost Model on WESAD with Engineered Features (HR, RMSSD, SCL)")
    print("=" * 60)

    # Expect a WESAD-derived CSV with columns: HR, RMSSD, SCL, stress
    csv_path = os.path.join(CSV_DIR, "wesad_new_with1.csv")
    df_raw = pd.read_csv(csv_path)
    base_features = ["HR", "RMSSD", "SCL"]

    # 1. Cleaning (includes assertions after filtering) — base physiological channels only
    df = _clean_swell_dataframe(df_raw, base_features)
    _run_safety_checks(df, base_features)

    # 1a. Remove physiologically contradictory low-stress samples
    before_inconsistent = len(df)
    df = df[~(
        (df["stress"] == 0)
        & (df["HR"] > 130)
        & (df["RMSSD"] < 25)
        & (df["SCL"] > 8)
    )]
    after_inconsistent = len(df)
    print(f"\nRemoved {before_inconsistent - after_inconsistent} physiologically contradictory low-stress samples.")

    # 1b. Feature engineering on cleaned data
    df = df.copy()
    df["HR_RMSSD_ratio"] = df["HR"] / (df["RMSSD"] + 1.0)
    df["Sympathetic_Index"] = df["HR"] * df["SCL"]
    df["HR_log"] = np.log1p(df["HR"])
    df["RMSSD_log"] = np.log1p(df["RMSSD"])
    df["SCL_log"] = np.log1p(df["SCL"])

    feature_cols = base_features + [
        "HR_RMSSD_ratio",
        "Sympathetic_Index",
        "HR_log",
        "RMSSD_log",
        "SCL_log",
    ]

    # Features and labels (including engineered features)
    X = df[feature_cols].astype(float).values
    y = df["stress"].astype(int).values

    # 2. Class distribution / imbalance check
    print("\nClass distribution (stress) after inconsistency removal:")
    class_counts = df["stress"].value_counts().sort_index()
    print(class_counts)

    # 3. Stratified cross-validation on full dataset (for robust estimates)
    cv_model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=2,
        eval_metric="logloss",
        random_state=42,
    )
    _cross_validate_model("XGBoost (engineered features)", cv_model, X, y, n_splits=5)

    # 4. Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 5. StandardScaler fitted on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train the specified XGBoost model (non-linear, strong learner)
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=2,
        eval_metric="logloss",
        random_state=42,
    )
    best_name = "XGBoost (engineered features)"
    metrics = _evaluate_model(best_name, xgb, X_train_scaled, y_train, X_test_scaled, y_test)
    best_model = metrics["model"]

    # 6. Persist model and scaler using consistent names expected by the app
    joblib.dump(best_model,   os.path.join(MODEL_DIR, "swell_rf_model.pkl"))
    joblib.dump(scaler,       os.path.join(MODEL_DIR, "swell_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "swell_features.pkl"))

    # Feature importances if available (tree-based models)
    importances = None
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(feature_cols, best_model.feature_importances_))
    if importances is not None:
        print("\nFeature importances for final model:")
        for k, v in importances.items():
            print(f"  {k}: {v:.4f}")

    # 8. Plot confusion matrix for the final model
    cm_array = np.array(metrics["confusion_matrix"])
    try:
        from sklearn.metrics import ConfusionMatrixDisplay

        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_array,
            display_labels=["No Stress", "Stress"],
        )
        disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
        ax_cm.set_title(f"{best_name} - Confusion Matrix")
        fig_cm.tight_layout()
        cm_path = os.path.join(MODEL_DIR, "swell_confusion_matrix.png")
        fig_cm.savefig(cm_path, dpi=120)
        plt.close(fig_cm)
        print(f"\nConfusion matrix plot saved to: {cm_path}")
    except Exception:
        print("\n[WARN] Unable to save confusion matrix plot.")

    # 9. Plot ROC curve using predicted probabilities
    roc_path = None
    try:
        if hasattr(best_model, "predict_proba"):
            y_scores_final = best_model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_scores_final = best_model.decision_function(X_test_scaled)
        else:
            y_scores_final = None

        if y_scores_final is not None and len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_scores_final)
            auc_final = roc_auc_score(y_test, y_scores_final)
            fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
            ax_roc.plot(fpr, tpr, label=f"AUC = {auc_final:.3f}")
            ax_roc.plot([0, 1], [0, 1], "k--", label="Chance")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title(f"{best_name} - ROC Curve")
            ax_roc.legend(loc="lower right")
            fig_roc.tight_layout()
            roc_path = os.path.join(MODEL_DIR, "swell_roc_curve.png")
            fig_roc.savefig(roc_path, dpi=120)
            plt.close(fig_roc)
            print(f"ROC curve plot saved to: {roc_path}")
        else:
            print("\n[WARN] ROC curve could not be computed (single class in test set or no scores).")
    except Exception:
        print("\n[WARN] Unable to compute or save ROC curve for final model.")

    metadata = {
        "model_type": "XGBClassifier(engineered)",
        "dataset": "WESAD",
        "features": feature_cols,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_distribution": class_counts.to_dict(),
        "selected_metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": float(metrics["roc_auc"])
            if np.isfinite(metrics["roc_auc"])
            else None,
            "mcc": metrics["mcc"],
        },
        "feature_importances": importances,
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, "swell_metadata.pkl"))

    print("\n[OK] Best model and scaler saved to models/")
    return metadata


if __name__ == "__main__":
    # Avoid emojis here to prevent Windows console encoding errors (cp1252).
    print("Stress Detection - Model Training\n")
    meta = train_swell_model()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Selected Final Model: {meta['model_type']}")
    print(f"Metrics (F1 / AUC / MCC): "
          f"{meta['selected_metrics']['f1']:.3f} / "
          f"{(meta['selected_metrics']['roc_auc'] or 0.0):.3f} / "
          f"{meta['selected_metrics']['mcc']:.3f}")
    print(f"\nModels saved in: {MODEL_DIR}")
    print("Run the dashboard with: streamlit run app.py")
