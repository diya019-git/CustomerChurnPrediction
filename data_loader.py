"""
data_loader.py
==============
Loads three publicly available churn datasets and harmonises them into one
unified feature schema so a single XGBoost model can be trained across all.

Datasets
--------
1. IBM Telco Customer Churn  (Kaggle: blastchar/telco-customer-churn)
   File : data/telco_churn.csv
   Rows : ~7,043

2. Orange Telecom Churn  (Kaggle: mnassrib/telecom-churn-datasets)
   Files: data/orange_churn_train.csv  +  data/orange_churn_test.csv
          (churn-bigml-80.csv + churn-bigml-20.csv, merged here)
   Rows : ~3,333

3. Iranian Churn Dataset  (UCI ML Repository – ID 563)
   File : data/iranian_churn.csv
   Rows : ~3,150

Unified Schema  (10 numeric features + binary target)
------------------------------------------------------
Feature                 Range / unit
---                     ---
tenure                  months with the company
monthly_charges         avg monthly payment (USD or normalized)
total_charges           cumulative charges to date
num_services            count of add-on services subscribed  (0–7)
has_internet            1 = has internet / data plan
has_phone               1 = has voice / phone plan
is_monthly_contract     1 = month-to-month / pay-as-you-go contract
cust_service_calls      # calls made to customer support
has_complaints          1 = customer has lodged a formal complaint
is_senior               1 = senior citizen / age-group ≥ 50
churn                   TARGET  1 = churned, 0 = stayed

Why a unified schema?
  Combining heterogeneous datasets enlarges the training distribution,
  exposes the model to diverse churn patterns (telecom, subscription),
  and reduces over-fitting to any single provider's characteristics.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset 1 – IBM Telco  (Kaggle: blastchar/telco-customer-churn)
# ──────────────────────────────────────────────────────────────────────────────

def load_telco(path: str = "data/telco_churn.csv") -> Optional[pd.DataFrame]:
    """
    Map the Telco dataset's rich feature set onto the unified schema.

    Mapping rationale
    -----------------
    num_services    = count of optional services marked 'Yes'
                      (MultipleLines, OnlineSecurity, OnlineBackup,
                       DeviceProtection, TechSupport, StreamingTV, StreamingMovies)
    has_internet    = 1 if InternetService != 'No'
    is_monthly_contract = 1 if Contract == 'Month-to-month'
    cust_service_calls  = 0  (Telco dataset doesn't record this)
    has_complaints      = 0  (Telco dataset doesn't record this)
    is_senior           = SeniorCitizen (already 0/1)
    """
    if not os.path.exists(path):
        print(f"[data_loader] Telco dataset not found at '{path}' — skipping.")
        return None

    df = pd.read_csv(path)
    df.drop(columns=["customerID"], errors="ignore", inplace=True)

    # Fix TotalCharges: some new customers (tenure=0) have blank strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Count add-on services that are explicitly 'Yes'
    addon_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    num_services = (df[addon_cols] == "Yes").sum(axis=1)

    unified = pd.DataFrame({
        "tenure":               df["tenure"].astype(float),
        "monthly_charges":      df["MonthlyCharges"].astype(float),
        "total_charges":        df["TotalCharges"].astype(float),
        "num_services":         num_services.astype(float),
        "has_internet":         (df["InternetService"] != "No").astype(int),
        "has_phone":            (df["PhoneService"] == "Yes").astype(int),
        "is_monthly_contract":  (df["Contract"] == "Month-to-month").astype(int),
        "cust_service_calls":   0,   # not captured → model will learn 0 is uninformative here
        "has_complaints":       0,   # not captured
        "is_senior":            df["SeniorCitizen"].astype(int),
        "churn":                (df["Churn"] == "Yes").astype(int),
    })
    print(f"[data_loader] Telco     : {len(unified):>5} rows  |  churn rate {unified['churn'].mean():.1%}")
    return unified


# ──────────────────────────────────────────────────────────────────────────────
# Dataset 2 – Orange Telecom  (Kaggle: mnassrib/telecom-churn-datasets)
# ──────────────────────────────────────────────────────────────────────────────

def load_orange(
    train_path: str = "data/orange_churn_train.csv",
    test_path:  str = "data/orange_churn_test.csv",
) -> Optional[pd.DataFrame]:
    """
    Merge the 80 / 20 split files and map onto the unified schema.

    Mapping rationale
    -----------------
    tenure              = 'Account length'  (months of account history)
    monthly_charges     = sum of all per-period charge columns
                          (day + evening + night + international)
    total_charges       = account_length × monthly_charges / 12
                          (proportional estimate; no exact total in raw data)
    num_services        = 1 if international plan + 1 if voice mail plan
    has_internet        = 0  (this is a voice-only telecom dataset)
    is_monthly_contract = 1  (account-based billing → treated as rolling)
    cust_service_calls  = 'Customer service calls'
    has_complaints      = 1 if cust_service_calls ≥ 4
                          (industry proxy: ≥4 calls signals serious dissatisfaction)
    is_senior           = 0  (age data not available)
    """
    frames = []
    for path in [train_path, test_path]:
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
        else:
            print(f"[data_loader] Orange file not found at '{path}' — skipping.")

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    # Normalise column names (strip spaces, lower-case, snake_case)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_", regex=False)
    )

    monthly = (
        df["total_day_charge"]
        + df["total_eve_charge"]
        + df["total_night_charge"]
        + df["total_intl_charge"]
    )
    tenure    = df["account_length"].astype(float)
    csc       = df["customer_service_calls"].astype(float)

    # Churn column might be bool (True/False) or string
    raw_churn = df["churn"]
    if raw_churn.dtype == bool:
        churn_int = raw_churn.astype(int)
    else:
        churn_int = raw_churn.map(
            {True: 1, False: 0, "True": 1, "False": 0, "yes": 1, "no": 0}
        ).fillna(0).astype(int)

    unified = pd.DataFrame({
        "tenure":               tenure,
        "monthly_charges":      monthly.astype(float),
        "total_charges":        (tenure * monthly / 12).astype(float),
        "num_services":         (
            (df["international_plan"].str.lower() == "yes").astype(int)
            + (df["voice_mail_plan"].str.lower() == "yes").astype(int)
        ).astype(float),
        "has_internet":         0,
        "has_phone":            1,   # all customers are voice subscribers
        "is_monthly_contract":  1,   # rolling monthly accounts
        "cust_service_calls":   csc,
        "has_complaints":       (csc >= 4).astype(int),
        "is_senior":            0,
        "churn":                churn_int,
    })
    print(f"[data_loader] Orange    : {len(unified):>5} rows  |  churn rate {unified['churn'].mean():.1%}")
    return unified


# ──────────────────────────────────────────────────────────────────────────────
# Dataset 3 – Iranian Churn  (UCI ML Repository – ID 563)
# ──────────────────────────────────────────────────────────────────────────────

def load_iranian(path: str = "data/iranian_churn.csv") -> Optional[pd.DataFrame]:
    """
    Map the UCI Iranian Churn dataset onto the unified schema.

    Raw columns (headers may contain extra spaces — normalised below):
      Call Failure, Complains, Subscription Length, Charge Amount,
      Seconds of Use, Frequency of use, Frequency of SMS,
      Distinct Called Numbers, Age Group, Tariff Plan, Status,
      Age, Customer Value, Churn

    Mapping rationale
    -----------------
    tenure              = Subscription Length  (months)
    monthly_charges     = Charge Amount × 15   (original scale 0–9 → multiply to
                          place in roughly same range as the other datasets)
    total_charges       = tenure × monthly_charges
    num_services        = 1 if Frequency of SMS > 0  (proxy for data/SMS plan)
    has_internet        = 0  (no internet-plan column)
    has_phone           = 1 if Frequency of use > 0  (active voice user)
    is_monthly_contract = 1 if Tariff Plan == 1  (pay-as-you-go → rolling)
    cust_service_calls  = Complains  (0 or 1; treat as binary support-contact flag)
    has_complaints      = Complains
    is_senior           = 1 if Age Group ≥ 4  (groups 4–5 ≈ 55+ age band)
    """
    if not os.path.exists(path):
        print(f"[data_loader] Iranian dataset not found at '{path}' — skipping.")
        return None

    df = pd.read_csv(path)

    # Normalise column names: strip whitespace, lower, snake_case
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"\s+", "_", regex=True)
    )

    # Accept minor column-name variants (UCI vs Kaggle versions may differ)
    col = lambda *candidates: next(
        (c for c in candidates if c in df.columns), None
    )

    tenure_col   = col("subscription_length", "subscription__length")
    charge_col   = col("charge_amount", "charge__amount")
    sms_col      = col("frequency_of_sms", "freq_of_sms")
    freq_col     = col("frequency_of_use", "frequency_of_use")
    age_grp_col  = col("age_group", "agegroup")
    tariff_col   = col("tariff_plan", "tariffplan")
    complain_col = col("complains", "complain")
    churn_col    = col("churn")

    tenure   = df[tenure_col].astype(float)
    charge   = df[charge_col].astype(float) * 15.0   # rescale to ≈ USD range

    unified = pd.DataFrame({
        "tenure":               tenure,
        "monthly_charges":      charge,
        "total_charges":        (tenure * charge).astype(float),
        "num_services":         (df[sms_col] > 0).astype(float) if sms_col else 0.0,
        "has_internet":         0,
        "has_phone":            (df[freq_col] > 0).astype(int) if freq_col else 1,
        "is_monthly_contract":  (df[tariff_col] == 1).astype(int) if tariff_col else 0,
        "cust_service_calls":   df[complain_col].astype(float) if complain_col else 0.0,
        "has_complaints":       df[complain_col].astype(int) if complain_col else 0,
        "is_senior":            (df[age_grp_col] >= 4).astype(int) if age_grp_col else 0,
        "churn":                df[churn_col].astype(int),
    })
    print(f"[data_loader] Iranian   : {len(unified):>5} rows  |  churn rate {unified['churn'].mean():.1%}")
    return unified


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

#: Ordered list of feature columns (excludes target 'churn')
FEATURE_COLS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_services",
    "has_internet",
    "has_phone",
    "is_monthly_contract",
    "cust_service_calls",
    "has_complaints",
    "is_senior",
]

#: Human-readable labels for the prediction form
FEATURE_LABELS = {
    "tenure":               "Tenure (months)",
    "monthly_charges":      "Monthly Charges ($)",
    "total_charges":        "Total Charges ($)",
    "num_services":         "Number of Add-on Services",
    "has_internet":         "Has Internet / Data Plan",
    "has_phone":            "Has Phone / Voice Plan",
    "is_monthly_contract":  "Month-to-Month Contract",
    "cust_service_calls":   "Customer Service Calls",
    "has_complaints":       "Has Formal Complaints",
    "is_senior":            "Senior Citizen",
}


def load_all_datasets() -> pd.DataFrame:
    """
    Load all available datasets, concatenate, and return a clean DataFrame.

    At least ONE dataset must be present.  Missing datasets are skipped with
    a warning so the app still runs if only one file has been downloaded.
    """
    frames = []
    loaders = [
        ("Telco",   load_telco),
        ("Orange",  load_orange),
        ("Iranian", load_iranian),
    ]

    for name, loader in loaders:
        try:
            result = loader()
            if result is not None and len(result) > 0:
                frames.append(result)
        except Exception as exc:
            print(f"[data_loader] {name} failed: {exc}")

    if not frames:
        raise FileNotFoundError(
            "No dataset files found in data/.  "
            "Please download at least one dataset — see DATASET.md for instructions."
        )

    combined = pd.concat(frames, ignore_index=True)

    # Drop rows with NaN in any feature or target column
    before = len(combined)
    combined.dropna(subset=FEATURE_COLS + ["churn"], inplace=True)
    dropped = before - len(combined)
    if dropped:
        print(f"[data_loader] Dropped {dropped} rows with missing values.")

    # Ensure correct dtypes
    for col in FEATURE_COLS:
        combined[col] = combined[col].astype(float)
    combined["churn"] = combined["churn"].astype(int)

    print(
        f"[data_loader] Combined  : {len(combined):>5} rows  |  "
        f"churn rate {combined['churn'].mean():.1%}  |  "
        f"{combined['churn'].sum()} positives"
    )
    return combined
