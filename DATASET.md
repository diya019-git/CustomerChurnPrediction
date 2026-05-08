# Dataset Download Instructions

The project trains on **three separate churn datasets** that are harmonised into
a single unified feature schema at runtime.  You can download any subset — the
app gracefully skips datasets whose files are missing.

---

## Dataset 1 — IBM Telco Customer Churn

| | |
|---|---|
| **Source** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **Rows** | 7,043 |
| **Target file** | `data/telco_churn.csv` |

### Download

**Option A — Kaggle Web UI**
1. Visit the URL above and click **Download**
2. Unzip and rename the file to `telco_churn.csv`
3. Place it in the `data/` folder

**Option B — Kaggle CLI**
```bash
kaggle datasets download -d blastchar/telco-customer-churn --unzip -p data/
mv data/WA_Fn-UseC_-Telco-Customer-Churn.csv data/telco_churn.csv
```

### Key Columns
`gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges, TotalCharges, Churn`

---

## Dataset 2 — Orange Telecom Churn

| | |
|---|---|
| **Source** | Kaggle |
| **URL** | https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets |
| **Rows** | 3,333 (80% train + 20% test files merged) |
| **Target files** | `data/orange_churn_train.csv`  +  `data/orange_churn_test.csv` |

### Download

**Option A — Kaggle Web UI**
1. Visit the URL above and click **Download**
2. Unzip — you will get `churn-bigml-80.csv` and `churn-bigml-20.csv`
3. Rename and move:
   - `churn-bigml-80.csv` → `data/orange_churn_train.csv`
   - `churn-bigml-20.csv` → `data/orange_churn_test.csv`

**Option B — Kaggle CLI**
```bash
kaggle datasets download -d mnassrib/telecom-churn-datasets --unzip -p data/
mv "data/churn-bigml-80.csv" data/orange_churn_train.csv
mv "data/churn-bigml-20.csv" data/orange_churn_test.csv
```

### Key Columns
`State, Account length, International plan, Voice mail plan,
Total day minutes/calls/charge, Total eve minutes/calls/charge,
Total night minutes/calls/charge, Total intl minutes/calls/charge,
Customer service calls, Churn`

---

## Dataset 3 — Iranian Churn Dataset

| | |
|---|---|
| **Source** | UCI ML Repository (ID: 563) |
| **UCI URL** | https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset |
| **Kaggle mirror** | https://www.kaggle.com/datasets/royjafari/customer-churn |
| **Rows** | 3,150 |
| **Target file** | `data/iranian_churn.csv` |

### Download

**Option A — UCI Direct**
1. Visit the UCI URL above
2. Click **Download** to get the zip
3. Extract and rename the CSV to `iranian_churn.csv`
4. Place it in `data/`

**Option B — Kaggle CLI**
```bash
kaggle datasets download -d royjafari/customer-churn --unzip -p data/
mv "data/Customer Churn.csv" data/iranian_churn.csv
```

### Key Columns
`Call Failure, Complains, Subscription Length, Charge Amount,
Seconds of Use, Frequency of use, Frequency of SMS,
Distinct Called Numbers, Age Group, Tariff Plan, Status, Age,
Customer Value, Churn`

---

## Expected Folder Layout After Download

```
data/
├── telco_churn.csv           ← Dataset 1
├── orange_churn_train.csv    ← Dataset 2 (train split)
├── orange_churn_test.csv     ← Dataset 2 (test split)
└── iranian_churn.csv         ← Dataset 3
```

---

## Unified Feature Schema

All three datasets are mapped to these 10 numeric features before training:

| Feature | Description |
|---|---|
| `tenure` | Months the customer has been with the company |
| `monthly_charges` | Average monthly payment (USD or normalized) |
| `total_charges` | Cumulative charges to date |
| `num_services` | Count of add-on services subscribed |
| `has_internet` | 1 = has internet / data plan |
| `has_phone` | 1 = has voice / phone plan |
| `is_monthly_contract` | 1 = month-to-month / pay-as-you-go contract |
| `cust_service_calls` | Number of calls to customer support |
| `has_complaints` | 1 = customer has lodged a formal complaint |
| `is_senior` | 1 = senior citizen / age-group ≥ 50 |
| **`churn`** | **TARGET — 1 = churned, 0 = stayed** |
