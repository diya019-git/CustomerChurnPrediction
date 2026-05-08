# Customer Churn Prediction Dashboard

A fully interactive machine-learning dashboard that predicts customer churn
using **XGBoost** trained on data augmented from **three real-world datasets**.
Built with Streamlit and an imbalanced-learning pipeline (SMOTE).

---

## Features

All 3 datasets were augumented based on common features to create a larger dataset to make the project more generalizable as compared to using a single dataset.
| Section | What it does |
|---|---|
| **Overview & EDA** | KPI cards, churn distribution, tenure/charge box-plots, correlation matrix, per-feature churn rates |
| **Model Performance** | Confusion matrix, ROC curve (with AUC), feature importance chart, full classification report |
| **Single Prediction** | Interactive form → churn probability gauge with risk level indicator |
| **Batch Prediction** | Upload CSV → scored table with risk bands → download results |

---

## Tech Stack

| Layer | Library |
|---|---|
| Model | XGBoost 2.x (`XGBClassifier`) |
| Preprocessing | scikit-learn `StandardScaler` |
| Imbalance handling | imbalanced-learn `SMOTE` |
| Dashboard | Streamlit |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Data | Pandas, NumPy |
| Environment | **uv** (fast Python package manager) |

---

## Datasets

Three publicly sourced datasets are harmonised into a single 10-feature schema:

| # | Dataset | Source | Rows |
|---|---|---|---|
| 1 | IBM Telco Customer Churn | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | 7,043 |
| 2 | Orange Telecom Churn | [Kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets) | 3,333 |
| 3 | Iranian Churn Dataset | [UCI ML Repository](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset) | 3,150 |

> See **DATASET.md** for full download instructions.

---

## Project Structure

```
.
├── app.py                  # Streamlit dashboard (4 pages)
├── train_model.py          # Training pipeline
├── data_loader.py          # Multi-dataset loading & feature harmonisation
├── pyproject.toml          # uv project manifest
├── requirements.txt        # pip-compatible deps
├── README.md               # This file
├── DATASET.md              # Dataset download instructions
├── data/                   # Place downloaded CSVs here
│   ├── telco_churn.csv
│   ├── orange_churn_train.csv
│   ├── orange_churn_test.csv
│   └── iranian_churn.csv
└── models/                 # Auto-created by train_model.py
    ├── xgb_model.pkl
    ├── scaler.pkl
    ├── metrics.json
    ├── feature_importance.csv
    └── feature_cols.json
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — install with `pip install uv` or `brew install uv`

### 1. Create virtual environment

```bash
uv venv .venv
```

### 2. Activate the environment

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

Or using `uv sync` (reads `pyproject.toml`):

```bash
uv sync
```

---

## Running the Project

### Step 1 — Download datasets

Follow the instructions in **DATASET.md** and place all CSV files into `data/`.

### Step 2 — Train the model

```bash
python train_model.py
```

Expected output:
```
============================================================
  Customer Churn Prediction — XGBoost Training
============================================================
[data_loader] Telco     :  7043 rows  |  churn rate 26.5%
[data_loader] Orange    :  3333 rows  |  churn rate 14.5%
[data_loader] Iranian   :  3150 rows  |  churn rate 15.6%
[data_loader] Combined  : 13526 rows  |  churn rate 19.5%  |  2635 positives

Train :  10820 rows  |  churn rate 19.5%
Test  :   2706 rows  |  churn rate 19.5%

After SMOTE — train rows : 17478  |  churn rate 50.0%
Best iteration : ...

Accuracy  : 0.xxxx
ROC-AUC   : 0.xxxx
F1 Score  : 0.xxxx
```

### Step 3 — Launch the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## ML Pipeline Explained

### Data Harmonisation (`data_loader.py`)
Each dataset has completely different columns (e.g. Telco has `Contract`, Orange has
`Customer service calls`, Iranian has `Subscription Length`). The `data_loader`
module maps each dataset's columns to a shared 10-feature schema:

```
tenure, monthly_charges, total_charges, num_services,
has_internet, has_phone, is_monthly_contract,
cust_service_calls, has_complaints, is_senior
```

Where a feature is unavailable in a dataset (e.g. `has_internet` for Orange which
is voice-only), it is set to a constant — XGBoost effectively learns to ignore
zero-variance columns for those rows.

### Preprocessing (`train_model.py`)
```
raw data  →  StandardScaler  →  SMOTE  →  XGBoost
```

- **StandardScaler** normalises all features to zero mean / unit variance,
  preventing features with large magnitudes (e.g. `total_charges`) from dominating
  the gradient updates.
- **SMOTE** (Synthetic Minority Over-sampling Technique) generates synthetic
  samples for the minority churn class in the **training set only**, so the model
  does not see a biased distribution. The test set is untouched.
- **XGBoost** is a gradient-boosted ensemble of decision trees. Key
  hyperparameters used:
  - `n_estimators=400` — number of boosting rounds
  - `max_depth=6` — maximum depth per tree
  - `learning_rate=0.05` — shrinks each tree's contribution (prevents overfitting)
  - `subsample=0.80` — row sampling per tree
  - `colsample_bytree=0.80` — feature sampling per tree
  - `early_stopping_rounds=30` — halts if validation loss stops improving

### Why Multiple Datasets?
Training on a single Telco dataset would produce a model biased to IBM's customer
demographics and product mix.  Augmenting with Orange (voice-heavy, US market)
and Iranian (subscription-based, different pricing scale) broadens the decision
boundary, reduces overfitting, and makes the model more generalisable across
different churn scenarios.

---

## Viva Quick-Reference

| Question | Answer |
|---|---|
| Why XGBoost? | Handles tabular data well, robust to outliers, built-in regularisation (gamma, alpha, lambda), fast with n_jobs=-1 |
| Why SMOTE? | ~27% churn rate → imbalanced. SMOTE creates synthetic minority samples by interpolating between existing ones, unlike random oversampling |
| Why StandardScaler? | XGBoost itself is scale-invariant but SMOTE's KNN step is distance-based and requires normalisation |
| Why stratified split? | Ensures the same churn ratio in train and test, avoiding a lucky/unlucky split |
| What is ROC-AUC? | Area Under the ROC Curve — measures how well the model ranks churners above non-churners regardless of threshold |
| What is the unified schema? | 10 numeric features extracted from all 3 datasets mapping conceptually equivalent signals to the same column names |
| What happens when a feature is missing for a dataset? | Set to 0 (constant). XGBoost assigns near-zero importance to zero-variance inputs from those rows |

---

## Submission Checklist

- [x] Streamlit app (`app.py`)
- [x] Training script (`train_model.py`)
- [x] Data loader with multi-dataset harmonisation (`data_loader.py`)
- [x] `requirements.txt`
- [x] Dataset download instructions (`DATASET.md`)
- [x] `README.md`
- [x] Virtual environment via `uv` (`pyproject.toml`)
