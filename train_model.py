"""
train_model.py
==============
Full training pipeline for the Customer Churn Prediction model.

Steps
-----
1. Load and harmonise all available datasets via data_loader.py
2. Stratified 80 / 20 train-test split
3. StandardScaler (all features are numeric after harmonisation)
4. SMOTE on training data only  →  address class imbalance
5. XGBoost classifier
6. Evaluate on held-out test set
7. Persist all artefacts to models/

Run from project root
---------------------
    python train_model.py
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_loader import FEATURE_COLS, load_all_datasets

warnings.filterwarnings("ignore")


def train() -> None:
    """End-to-end training run.  Saves all artefacts into models/."""

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Customer Churn Prediction — XGBoost Training")
    print("=" * 60)

    combined = load_all_datasets()

    X = combined[FEATURE_COLS].values
    y = combined["churn"].values

    # ── 2. Stratified split ───────────────────────────────────────────────
    # stratify=y ensures the churn ratio is preserved in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain : {len(X_train):>6} rows  |  churn rate {y_train.mean():.1%}")
    print(f"Test  : {len(X_test):>6} rows  |  churn rate {y_test.mean():.1%}")

    # ── 3. StandardScaler ─────────────────────────────────────────────────
    # Fit ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── 4. SMOTE ─────────────────────────────────────────────────────────
    # Applied AFTER splitting so test set remains untouched
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)
    print(f"\nAfter SMOTE — train rows : {len(X_train_res)}  |  churn rate {y_train_res.mean():.1%}")

    # ── 5. XGBoost ────────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.5,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test_sc, y_test)],
        verbose=False,
    )
    print(f"Best iteration : {model.best_iteration}")

    # ── 6. Evaluation ─────────────────────────────────────────────────────
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    print(f"\nAccuracy  : {acc:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    metrics = {
        "accuracy":  float(acc),
        "roc_auc":   float(auc),
        "f1":        float(f1),
        "confusion_matrix":        confusion_matrix(y_test, y_pred).tolist(),
        "classification_report":   classification_report(y_test, y_pred, output_dict=True),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
    }

    # ── Feature importance ─────────────────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # ── 7. Persist artefacts ───────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    importance_df.to_csv("models/feature_importance.csv", index=False)

    # Store column names so the app can rebuild DataFrames correctly
    with open("models/feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print("\nArtefacts saved:")
    for fname in ["scaler.pkl", "xgb_model.pkl", "metrics.json",
                  "feature_importance.csv", "feature_cols.json"]:
        print(f"  models/{fname}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
