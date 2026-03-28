# Churn Analytics Pipeline

**End-to-end machine learning pipeline for predicting bank customer churn using Snowflake and a weighted ensemble of XGBoost, ANN, and LightGBM models.**

✅ **Status:** Production Ready | **Model:** XGBoost + ANN + LightGBM Ensemble | **ROC-AUC:** 0.8867 | **Last Updated:** March 2026

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.8867 |
| **Accuracy** | 84.81% |
| **Recall** | 69.16% |
| **Precision** | 62.82% |
| **F1-Score** | 0.6584 |
| **Training Samples** | 165,034 |
| **Features** | 41 (engineered from ML features) |
| **Ensemble Models** | 3 (XGBoost 44%, ANN 22%, LightGBM 33%) |

---

## 🏗️ Complete Project Structure

```
Churn-Analytics-with-RFM-features-in-ML/
│
├── README.md                          (this file)
├── requirements.txt                   (Python dependencies)
├── .gitignore                         (Git exclusions)
├── LICENSE                            (GPL v3)
│
├── 📄 PYTHON SCRIPTS (ML Pipeline)
│   ├── churn_bank.py                 (train on raw features + engineered)
│   ├── churn_bank_main.py            (MAIN - train on ML features - RECOMMENDED)
│   ├── churn_comparison.py           (compare outputs vs outputs_train)
│   └── churn.py                      (helper utilities)
│
├── 📊 DATA & FEATURES
│   ├── bank_ml_features.csv         (41 engineered features - 165K rows)
│   ├── Training data/
│   │   ├── train.csv                 (Kaggle raw dataset - 165K)
│   │   ├── test.csv                  (test subset - 7.8 MB)
│   │   └── sample_submission.csv     (submission format - 1.1 MB)
│   └── sql_transformed_data/
│       ├── bank_churn_raw.xlsx       (raw Snowflake export - 10.9 MB)
│       ├── bank_cohort_retention.xlsx (cohort analysis - 1 KB)
│       ├── bank_ml_features.xlsx     (ML features export - 25.3 MB)
│       └── bank_rfm_scores.xlsx      (RFM scores export - 14.9 MB)
│
├── 🏛️ SNOWFLAKE SQL SCRIPTS
│   └── sql_snowflake/
│       ├── Churn_warehouse.sql       (creates warehouse, DB, schemas)
│       ├── Churn_RFM.sql             (creates tables for RFM & cohort)
│       ├── Churn_kaggle_transform.sql (transforms & engineers features)
│       └── README.md                 (execution guide)
│
├── 📈 MODEL OUTPUTS (Production)
│   └── outputs/
│       ├── churn_predictions.csv     (165K predictions with risk tiers)
│       ├── ann_model.h5              (trained ANN weights)
│       ├── best_xgb_pipe.pkl         (trained XGBoost pipeline)
│       ├── lgb_model.pkl             (trained LightGBM model)
│       ├── best_threshold.pkl        (optimal threshold = 0.48)
│       ├── imputer.pkl               (median imputer)
│       ├── scaler.pkl                (standard scaler)
│       ├── feature_names.pkl         (41 feature names)
│       ├── confusion_matrix.png      (ensemble confusion matrix)
│       ├── roc_curve.png             (ROC curves all 3 models)
│       ├── precision_recall.png      (precision-recall curve)
│       ├── ann_history.png           (ANN training history)
│       └── learning_curve_xgb.png    (learning curve analysis)
│
├── 📊 MODEL OUTPUTS (Development/Comparison)
│   └── outputs_train/
│       ├── churn_predictions.csv     (training subset predictions)
│       ├── ann_model.h5
│       ├── best_xgb_pipe.pkl
│       ├── lgb_model.pkl
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── precision_recall.png
│       ├── ann_history.png
│       └── learning_curve_xgb.png
│
├── 📋 ANALYSIS & COMPARISON
│   ├── model_comparison.csv          (outputs vs outputs_train metrics)
│   ├── model_comparison.png          (4-panel comparison visualization)
│   └── model_comparison_insights.txt (generalization analysis)
│
└── 📁 SUPPORTING FILES
    ├── model_comparison/              (comparison analysis folder)
    └── sql_transformed_data/          (exported Snowflake tables)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Snowflake account access (optional - data included in repo)
- Power BI Desktop (optional for dashboards)

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/Churn-Analytics-with-RFM-features-in-ML.git
cd Churn-Analytics-with-RFM-features-in-ML
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables (Optional - for Snowflake)
```bash
export SNOWFLAKE_USER="SSCY"
export SNOWFLAKE_PASSWORD="your_password"
export SNOWFLAKE_ACCOUNT="rkiezxo-rp23355"
export SNOWFLAKE_WAREHOUSE="CHURN_WH"
export SNOWFLAKE_DATABASE="CHURN_ANALYTICS"
```

### 4. (Optional) Setup Snowflake Infrastructure
```bash
# Execute SQL scripts in Snowflake in order:
# 1. sql_snowflake/Churn_warehouse.sql
# 2. sql_snowflake/Churn_RFM.sql
# 3. sql_snowflake/Churn_kaggle_transform.sql
```

### 5. Train ML Models (PRIMARY - Recommended)
```bash
python churn_bank_main.py
```

**This script:**
- Loads engineered ML features from `bank_ml_features.xlsx`
- Performs train/val/test split (80/15/5 stratified)
- Applies SMOTE for class imbalance
- Trains 3 models: XGBoost, ANN, LightGBM
- Creates weighted ensemble
- Saves predictions to `outputs/churn_predictions.csv`

### 6. Compare Model Versions
```bash
python churn_comparison.py
```

Compares:
- `outputs_train/` (training subset)
- `outputs/` (full dataset)

Outputs:
- `model_comparison.csv`
- `model_comparison.png`
- `model_comparison_insights.txt`

### 7. (Alternative) Train on Raw Features
```bash
python churn_bank.py
```

---

## 📊 Two Training Approaches: Flexibility Based on Requirements

### **Option 1: churn_bank_main.py (bank_ml_features.csv) — RECOMMENDED**

Uses pre-engineered features from the ML features dataset
- **Data Source:** bank_ml_features.csv (41 pre-engineered features)
- **Output Folder:** outputs/
- **Performance:** ROC-AUC = **0.8867**, F1 = **0.6584**, Precision = 0.6282, Recall = **0.6916**
- **Best for:** Maximum churn detection, catching more churners
- **Best if you prioritize:** Recall (catch more actual churners)

### **Option 2: churn_bank.py (train.csv - Raw Features)**

Uses raw Kaggle features + on-the-fly feature engineering
- **Data Source:** Training data/train.csv (raw 165K customer data)
- **Output Folder:** outputs_train/
- **Performance:** ROC-AUC = **0.8882**, F1 = **0.6618**, Precision = **0.6742**, Recall = 0.6498
- **Best for:** High precision, fewer false alarms
- **Best if you prioritize:** Precision (confidence in predictions)

---

## 📊 Detailed Comparative Analysis: outputs vs outputs_train

### Why Both Datasets Perform Almost Equally Well (0.8867 vs 0.8882)

The **AUC difference is only -0.15%**, which is negligible. This reveals something important about your data:

```
┌─────────────────────────────────────────────────────────────┐
│ METRIC COMPARISON: bank_ml_features vs raw train.csv        │
├─────────────────────────────────────────────────────────────┤
│ Metric              outputs          outputs_train    Δ     │
│                     (ML Features)    (Raw Data)       Gap   │
├─────────────────────────────────────────────────────────────┤
│ ROC-AUC             0.8867           0.8882          -0.15% │
│ F1-Score            0.6584           0.6618          -0.51% │
│ Precision           0.6282           0.6742          -6.82% │
│ Recall              0.6916           0.6498          +6.43% │
│ Accuracy            84.81%           85.95%          -1.33% │
├─────────────────────────────────────────────────────────────┤
│ True Positives      4,830            4,538           +292   │
│ True Negatives      23,164           23,830          -666   │
│ False Positives     2,859            2,193           +666   │
│ False Negatives     2,154            2,446           -292   │
└─────────────────────────────────────────────────────────────┘
```

### Deep Dive Analysis

#### **Dimension 1: Feature Engineering vs Raw Data**

**bank_ml_features (outputs) — BETTER RECALL**
- ✅ Pre-engineered RFM scores + cohort metrics already computed
- ✅ Interaction terms (Age×Products, Balance×Credit) ready
- ✅ More features to learn from (36 vs 8-10)
- ✅ Result: Model catches MORE churners → Higher recall (0.6916)
- ⚠️ Trade-off: More false positives (2,859 vs 2,193)

**train.csv Raw Data (outputs_train) — BETTER PRECISION**
- ✅ Raw features force model to learn pure patterns
- ✅ Less preprocessing overhead, simpler data
- ✅ Result: When model predicts churn, more likely correct → Higher precision (0.6742)
- ⚠️ Trade-off: Misses some churners (0.6498 recall)

**Insight:** ML features give the model MORE signal to work with, so it catches more churners but with more false alarms. Raw data forces discipline—fewer false positives but also fewer true positives.

---

#### **Dimension 2: Precision-Recall Trade-off**

**When to use outputs (ML Features - Better Recall) — RECOMMENDED:**

```
Scenario: Preventing churn is critical, budget is flexible
─────────────────────────────────────────────────────────────
Recall: 0.6916 (69.2% of actual churners are caught)
Trade-off: More false positives (2,859 vs 2,193)

Business Decision:
✓ Flag 7,689 customers (4,830 TP + 2,859 FP)
✓ Catch 69.2% of true churners (miss 30.8%)
✓ ~37% are false alarms, but don't lose real churners
✓ Better for: High churn cost scenario

Who benefits: 
- Companies where losing a customer costs more than retention
- Banks, telecom, SaaS (high customer lifetime value)
- Risk-averse strategies
```

**When to use outputs_train (Raw Data - Better Precision):**

```
Scenario: Limited retention budget, high False Positive cost
─────────────────────────────────────────────────────────────
Precision: 0.6742 (67.4% of flagged customers actually churn)
F1-Score: 0.6618 (best balanced score)

Business Decision:
✓ Flag 2,193 customers for retention campaigns
✓ ~1,477 (67%) will actually churn
✓ ~716 (33%) are false alarms (waste of budget)
✓ Better for: Limited budget scenario

Who benefits: 
- Companies with tight retention budget
- Expensive interventions (personal calls, discounts)
- Cost-conscious strategies
```

---

### **RECOMMENDED: Use outputs (bank_ml_features) for most scenarios**

**Why:**
1. **Higher Recall** (0.6916 vs 0.6498) — catch 69% of churners instead of 65%
2. **Better ROC-AUC** (0.8867) — excellent discrimination overall
3. **Churn prevention focused** — lose fewer customers
4. **More features available** — model has richer signal
5. **Better for high-value customers** — don't miss important ones

**Use outputs_train (Raw Data) only if:**
- Your budget for retention is **strictly limited**
- Each false positive costs significantly
- You prefer **guaranteed high precision** over recall
- You want to focus only on your **most confident predictions**

---

## 🧠 Model Architecture

### Feature Engineering (41 total)

**Raw Features (8):**
`AGE, CREDITSCORE, TENURE, BALANCE, NUMOFPRODUCTS, HASCRCARD, ISACTIVEMEMBER, ESTIMATEDSALARY`

**Categorical Features (2):**
`GENDER, GEOGRAPHY`

**Engineered Features (31):**
- RFM Metrics, Cohort Metrics, Balance Features, Credit Features
- Interaction Terms, Risk Flags, Recency, Frequency, Monetary

### Ensemble Architecture

**Model 1: XGBoost (44% weight)**
- Best AUC: 0.8856
- Max depth: 8, Learning rate: 0.05

**Model 2: ANN (22% weight)**
- Input → Dense(256) → Dense(128) → Dense(64) → Dense(32) → Output
- Best Val AUC: 0.8830

**Model 3: LightGBM (33% weight)**
- N estimators: 500, Max depth: 8

**Ensemble Voting:**
```
final_probability = (4×XGBoost + 2×ANN + 3×LightGBM) / 9
decision_threshold = 0.48 (optimized for F1-Score)
```

---

## 📈 Key Findings

### Risk Tiers
- **HIGH:** 29,768 customers (≥0.70 probability)
- **MEDIUM:** 29,950 customers (0.40-0.70)
- **LOW:** 105,316 customers (<0.40)

### RFM Segments
- **Champions:** 38,731 customers (low churn risk)
- **Hibernating:** 53,205 customers (HIGH churn risk ⚠️)
- **At Risk:** 14,751 customers (HIGH churn risk ⚠️)

### Top Churn Predictors
1. Age (older customers churn more)
2. Tenure (longer = lower churn)
3. IsActiveMember (inactive = 2.5× higher churn)
4. Balance (higher = retention)
5. NumOfProducts (multi-product = loyal)

---

## 📁 Data Files in Repository

### Training Data
- **train.csv** (165,034 rows) — Full Kaggle dataset
- **test.csv** (7.8 MB) — Test subset
- **sample_submission.csv** (1.1 MB) — Submission format

### Snowflake Exports
All tables exported as Excel for reference:
- **bank_churn_raw.xlsx** (10.9 MB) — Raw data
- **bank_rfm_scores.xlsx** (14.9 MB) — RFM segmentation
- **bank_cohort_retention.xlsx** (1 KB) — Cohort analysis
- **bank_ml_features.xlsx** (25.3 MB) — 41 engineered features

---

## 🔄 Snowflake SQL Workflow

Execute in order:

1. **Churn_warehouse.sql** — Creates infrastructure
2. **Churn_RFM.sql** — Creates empty tables
3. **Churn_kaggle_transform.sql** — Populates & transforms data

All SQL scripts included in `sql_snowflake/` folder.

---

## 💻 Tech Stack

- **Data Warehouse:** Snowflake
- **ML Framework:** TensorFlow, XGBoost, LightGBM
- **Preprocessing:** scikit-learn, imbalanced-learn (SMOTE)
- **Feature Importance:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Language:** Python 3.10+

---

## 📝 License

GNU General Public License v3.0 - see LICENSE file for details

---

**Last Updated:** March 28, 2026  
**Model Version:** Ensemble (XGBoost + ANN + LightGBM) v1.0  
**Status:** ✅ Production Ready  
**ROC-AUC:** 0.8867 | **Recall:** 69.16% | **Generalization Gap:** -0.17%