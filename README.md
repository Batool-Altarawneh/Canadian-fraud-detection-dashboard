# Canadian Fraud Detection Dashboard

End-to-end fraud analytics pipeline built on a Canadianized version of the Kaggle Credit Card Fraud dataset. The project covers data engineering, star schema design, SQL Server, machine learning, and a Power BI dashboard , all connected into a single deployable system.

---

## Dashboard Preview

| Page | Description |
|------|-------------|
| Home | Project landing page with key stats and navigation |
| Executive Overview | KPIs, monthly fraud trend, fraud by type, loss by province, Fraud Rate % by Age Group |
| Fraud Trends | Day × hour heatmap, rolling 30-day trend, peak hours, category breakdown |
| Geo Analysis | Province map, top cities by fraud volume, province summary table |
| Transaction Drill-Through | Row-level transaction detail with customer and merchant info |
| ML Model Performance | XGBoost results, SHAP feature importance, risk tier breakdown |




### Home
![Home](powerbi/Images/Home.png)
 
### Executive Overview
![Executive Overview](powerbi/Images/Executive%20Overview.png)
 
### Fraud Trends
![Fraud Trends](powerbi/Images/Trends.png)
 
### Geo Analysis
![Geo Analysis](powerbi/Images/Geo%20Analysis.png)
 
### Transaction Drill-Through
![Transaction Drill-Through](powerbi/Images/Drill-Through.png)
 
### ML Model Performance
![ML Model Performance](powerbi/Images/ML%20Model.png)
 

---

## Project Stats

| Metric | Value |
|--------|-------|
| Total transactions | 1,296,675 |
| Fraud cases | 7,506 |
| Fraud rate | .5789% |
| Date range | 2019–2020 |
| Provinces | 8 (ON, AB, BC, MB, SK, NS, NB, QC) |
| ML model PR-AUC | 0.9105 |
| ML net benefit | $966,074 CAD (test set) |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data source | Kaggle CC Fraud Dataset (CC0), Canadianized synthetic data |
| ETL | Python, pandas, geopy |
| Feature engineering | pandas, numpy, scikit-learn |
| Database | SQL Server (FraudDB), SQLAlchemy |
| Schema | Star schema — 4 dimension tables + 1 fact table + MLPredictions |
| ML | scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, joblib |
| Dashboard | Power BI Desktop |
| Environment | Python 3.10, virtual environment |

---

## Repository Structure

```
canadian-fraud-detection-dashboard/
│
├── data/
│   ├── raw/                        # Original Kaggle CSV files (gitignored)
│   └── processed/                  # Cleaned dimension and fact CSVs
│       ├── DimCustomer.csv
│       ├── DimMerchant.csv
│       ├── DimDate.csv
│       ├── DimAlertType.csv
│       ├── FactTransaction.csv
│       ├── fraud_features.csv      # ML-ready feature set
│       ├── ml_predictions.csv      # Scored predictions (1.29M rows)
│       ├── ml_model_comparison.csv # Model comparison for Power BI
│       ├── ml_threshold_curve.csv  # Threshold tuning curve for Power BI
│       └── ml_shap_importance.csv  # SHAP feature importance for Power BI
│
├── etl/
│   ├── etl_pipeline.py             # Raw data ingestion and Canadianization
│   ├── feature_engineering.py      # fraud_score, risk_tier, age_group, amount_zscore
│   ├── dim_builder.py              # Builds 4 dimension tables from features
│   └── sql_loader.py               # Loads CSVs into SQL Server FraudDB
│
├── ml/
│   ├── fraud_detection_ml.ipynb    # Full 5-step ML pipeline notebook
│   ├── models/
│   │   ├── best_model.pkl          # Final XGBoost model
│   │   ├── model_logreg.pkl        # Baseline logistic regression
│   │   ├── model_random_forest.pkl # Random Forest model
│   │   ├── model_xgboost.pkl       # XGBoost model
│   │   ├── model_metadata.json     # Model card with all training details
│   │   └── label_encoders.pkl      # Saved LabelEncoders for inference
│   └── reports/
│       ├── step1_baseline_logreg.png
│       ├── step2_random_forest.png
│       ├── step2_xgboost.png
│       ├── step2_model_comparison.png
│       ├── step3a_threshold_tuning.png
│       ├── step3b_confusion_matrices.png
│       ├── step4a_shap_summary.png
│       ├── step4b_shap_importance.png
│       ├── step4c_shap_waterfall.png
│       └── step5a_score_distribution.png
│
├── notebooks/
│   └── explore.ipynb               # Initial data exploration
│
├── powerbi/
│   └── FraudDB.pbix                # Power BI report file
│
├── sql/
│   ├── Create_FraudDB.sql          # Creates the FraudDB database
│   ├── create_tables.sql           # Star schema DDL (5 tables)
│   ├── stored_procs.sql            # usp_GetFraudSummaryByProvince
│   ├── analytical_queries.sql      # 5 analytical queries
│   └── SSMS_Verification_Command.sql
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Database Schema

```
DimCustomer ──────────────┐
DimMerchant ──────────────┤
DimDate ──────────────────┼──► FactTransaction ◄──── MLPredictions
DimAlertType ─────────────┘         (trans_num FK)
```

**FactTransaction** — 1,296,675 rows, one per transaction  
**MLPredictions** — 1,296,675 rows, one per transaction, joined on `trans_num`

---

## ML Pipeline

The notebook `ml/fraud_detection_ml.ipynb` runs end to end in 5 steps:

### Step 1 — Baseline
- Logistic Regression with `class_weight='balanced'`
- Time-aware 80/20 chronological split (no data leakage)
- PR-AUC: **0.1625**

### Step 2 — Model Comparison
- SMOTE resampling (`sampling_strategy=0.3`) on training set only
- Random Forest: PR-AUC **0.7173**
- XGBoost with `scale_pos_weight=171`: PR-AUC **0.911** ← selected

### Step 3 — Threshold Tuning
- Scanned 0.01 → 0.90, maximised F2-Score (recall weighted 2×)
- F2-optimal threshold: **0.890** (F2 = 0.8475, alerts = 1,984)
- Business threshold: **0.50** (net benefit = $966,078 CAD, alerts = 4,372)
- `FINAL_THRESHOLD = 0.50` selected for deployment

### Step 4 — SHAP Explainability
- `TreeExplainer` on XGBoost using stratified SHAP sample
- Top fraud drivers: `amount_cad`, `category`, `transaction_hour`
- Waterfall plot for highest-risk transaction (prob = 0.9999)

### Step 5 — Deployment
- Saved `best_model.pkl` + `model_metadata.json`
- Scored all 1,296,675 transactions with `ml_fraud_prob`, `ml_fraud_flag`, `ml_risk_tier`, `is_test_set`
- Loaded into `dbo.MLPredictions` in SQL Server
- Exported 3 CSVs for Power BI Page 6

---

## Setup

### Prerequisites
- Python 3.10+
- SQL Server with ODBC Driver 17
- Power BI Desktop

### Installation

```bash
git clone https://github.com/Batool-Altarawneh/Canadian-fraud-detection-dashboard.git
cd canadian-fraud-detection-dashboard
python -m venv .venv
.venv\Scripts\activate       
pip install -r requirements.txt
```

### Environment variables

Create `etl/.env`:

```
SQL_SERVER=localhost\SQLEXPRESS
SQL_DATABASE=FraudDB
```

### Running the pipeline

```bash
# 1. Create the database
# Run sql/Create_FraudDB.sql in SSMS

# 2. Create the star schema tables
# Run sql/create_tables.sql in SSMS

# 3. Run ETL
python etl/etl_pipeline.py
python etl/feature_engineering.py
python etl/dim_builder.py

# 4. Load into SQL Server
python etl/sql_loader.py

# 5. Run ML pipeline
# Open ml/fraud_detection_ml.ipynb and run all cells

# 6. Open Power BI
# Open powerbi/FraudDB.pbix
# Refresh data source connection
```

---

## Key Results

| Model | PR-AUC | ROC-AUC | F2-Score |
|-------|--------|---------|----------|
| Logistic Regression | 0.1625 | 0.9182 | 0.2849 |
| Random Forest + SMOTE | 0.7173 |  0.9938 | 0.5991 |
| **XGBoost** | **0.9105** | **0.9983** | **0.7046** |

**At FINAL_THRESHOLD = 0.50 (test set):**

| Metric | Value |
|--------|-------|
| Precision | 33.92% |
| Recall | 96.42% |
| True Positives | 1,483 |
| False Negatives | 55 |
| Net Benefit CAD | $966,078 |

---

## Data Source

Kaggle Credit Card Fraud Detection dataset (CC0 Public Domain).  
Customer names, addresses, merchants, and locations were replaced with Canadian synthetic equivalents. No real personal data is used.

---
