# Churn Analytics Pipeline

**End-to-end machine learning pipeline for predicting bank customer churn using Snowflake and ensemble ML models.**

✅ **Status:** Production Ready | **Model:** ANN + XGBoost Ensemble | **ROC-AUC:** 0.8867 | **Last Updated:** March 2026

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
| **Features** | 36 (engineered) |

---

## 🏗️ Architecture Overview

```
Kaggle Bank Dataset (165K rows)
           ↓
Snowflake Data Warehouse
├── PUBLIC.BANK_CHURN_RAW (raw data)
├── ANALYTICS.BANK_RFM_SCORES
├── ANALYTICS.BANK_COHORT_RETENTION
├── ANALYTICS.BANK_ML_FEATURES (36 features)
└── ANALYTICS.CHURN_SCORES (predictions)
           ↓
Python ML Pipeline (PRIMARY)
├── ANN (4 layers: 256→128→64→32)
├── XGBoost (RandomizedSearchCV tuned)
└── Ensemble Voting Classifier
           ↓
Power BI + Tableau Dashboards
```

**Optional:** Azure Data Lake Storage Gen2 + Azure Data Factory for automated daily pipeline orchestration (see Deployment section)

---

## 📁 Repository Structure

```
churn-analytics/
│
├── README.md                      (this file)
├── requirements.txt               (Python dependencies)
│
├── sql/                          (Snowflake SQL scripts)
│   ├── churn_warehouse.sql       (schema + tables)
│   ├── churn_rfm.sql             (RFM scoring)
│   ├── churn_fix_mlfeatures.sql  (feature engineering)
│   └── churn_kaggle_transform.sql (data transformation)
│
├── ml/                           (Python ML scripts - CORE)
│   ├── churn_bank.py             (main training script)
│   ├── model_comparison.py       (outputs vs outputs_train)
│   ├── snowflake_writeback.py    (predictions writeback)
│   └── utils.py                  (helper functions)
│
├── dashboards/
│   ├── powerbi_measures.dax      (DAX formulas for Power BI)
│   └── tableau_workbook.twbx     (Tableau template)
│
├── outputs/                      (model artifacts & results)
│   ├── churn_predictions.csv
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall.png
│   ├── shap_importance.png
│   ├── ann_model.h5
│   ├── best_xgb_pipe.pkl
│   └── best_threshold.pkl
│
└── .github/workflows/            (CI/CD - optional)
    └── model_retraining.yml
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Snowflake account access
- Power BI Desktop (optional)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/churn-analytics.git
cd churn-analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
export SNOWFLAKE_USER="SSCY"
export SNOWFLAKE_PASSWORD="your_password"
export SNOWFLAKE_ACCOUNT="rkiezxo-rp23355"
export SNOWFLAKE_WAREHOUSE="CHURN_WH"
export SNOWFLAKE_DATABASE="CHURN_ANALYTICS"
```

### 4. Create Snowflake Schema
Execute SQL files in order in your Snowflake account:
```sql
-- 1. Create base tables and raw data
source sql/churn_warehouse.sql

-- 2. Create RFM scores
source sql/churn_rfm.sql

-- 3. Create ML features
source sql/churn_fix_mlfeatures.sql

-- 4. Transform and load data
source sql/churn_kaggle_transform.sql
```

### 5. Train ML Model (PRIMARY)
```bash
python ml/churn_bank.py
```
**Outputs:**
- Model artifacts (ANN + XGBoost)
- `churn_predictions.csv` (predictions on full dataset)
- Visualizations (confusion matrix, ROC curve, SHAP importance, etc.)
- Saved to `outputs/` folder

### 6. Compare Model Versions
```bash
python ml/model_comparison.py
```
**Outputs:**
- `model_comparison.csv` (side-by-side metrics)
- `model_comparison.png` (4-panel visualization)
- `model_comparison_insights.txt` (detailed analysis)

### 7. Write Predictions to Snowflake
```bash
python ml/snowflake_writeback.py
```
Loads predictions to `CHURN_SCORES` table for dashboard consumption

### 8. Build Dashboards
**Power BI:**
- Open Power BI Desktop
- Get Data → Snowflake
- Connect to `CHURN_ANALYTICS` database
- Import all 5 tables (BANK_CHURN_RAW, RFM_SCORES, COHORT_RETENTION, ML_FEATURES, CHURN_SCORES)
- Add DAX measures from `dashboards/powerbi_measures.dax`
- Create visuals:
  - KPI card: Total customers (165K)
  - Bar chart: Churn by geography
  - Pie chart: Customer distribution by exit status
  - Matrix: RFM segmentation with frequency/monetary
  - Line chart: Cohort retention over time
  - Scatter plot: Age vs Balance by churn status

**Tableau:**
- Open `dashboards/tableau_workbook.twbx`
- Refresh data connection to Snowflake

---

## 🧠 Model Details

### Features (36 total)

**Raw Features (10):**
- CreditScore, Age, Tenure, Balance, EstimatedSalary
- NumOfProducts, HasCrCard, IsActiveMember, Geography, Gender

**Engineered Features (26):**
- RFM scores (R_SCORE, F_SCORE, M_SCORE, RFM_TOTAL)
- Cohort metrics (COHORT_AVG_AGE, COHORT_SIZE, COHORT_CHURN_RATE)
- Interaction terms (AGE_X_PRODUCTS, BALANCE_X_SALARY, CREDIT_X_ACTIVE)
- Risk flags (SENIOR_INACTIVE, LOW_RFM_FLAG)

### Model Architecture

**ANN:**
```
Input (36) → Dense(256, ReLU) → Dense(128, ReLU) 
           → Dense(64, ReLU) → Dense(32, ReLU)
           → Output(1, Sigmoid)
Optimizer: Adam | Loss: Binary Crossentropy | Epochs: 25
```

**XGBoost:**
```
max_depth: 5
learning_rate: 0.05
n_estimators: tuned via RandomizedSearchCV
scale_pos_weight: 4.0 (class weight balancing)
```

**Ensemble:**
```
VotingClassifier(
    estimators=[ANN, XGBoost],
    voting='soft',
    weights=[0.5, 0.5]
)
```

### Imbalance Handling
- **SMOTE:** Oversampling minority class (sampling_strategy=1.0)
- **Stratified K-Fold:** 5-fold with shuffled splits
- **Class Weights:** Penalize false negatives (missed churners)

### Cross-Validation
- Stratified K-Fold (5 splits)
- Repeated K-Fold (10 repeats)
- RandomizedSearchCV (100 iterations)
- Learning curves plotted for analysis

---

## 📊 Key Findings

### Churn Distribution
- **Total Customers:** 165,034
- **Churned:** 32,864 (19.9%)
- **Retained:** 132,170 (80.1%)

### Risk Tiers
| Tier | Count | Prediction Threshold |
|------|-------|----------------------|
| **HIGH** | 29,768 | ≥ 0.70 |
| **MEDIUM** | 29,950 | 0.40 - 0.70 |
| **LOW** | 105,316 | < 0.40 |

### RFM Segments
| Segment | Count | Avg Frequency | Avg Monetary | Churn Risk |
|---------|-------|---|---|---|
| **Champions** | 38,731 | 143K | $59M | Low |
| **Loyal** | 30,609 | 100K | $68M | Low |
| **Hibernating** | 53,205 | 99K | $75M | **HIGH** ⚠️ |
| **At Risk** | 14,751 | 46K | $55M | **HIGH** ⚠️ |
| **Needs Attention** | 18,195 | 52K | $57M | Medium |
| **Recent** | 9,543 | 21K | $82M | Low |

### Geographic Insights
| Country | Churn Rate | Avg Age | Avg Tenure |
|---------|-----------|---------|------------|
| **Germany** | 32% | 45 | 5.1 yrs |
| **Spain** | 17% | 37 | 7.3 yrs |
| **France** | 16% | 38 | 7.5 yrs |

### Top Churn Predictors (SHAP)
1. Age (strong positive correlation)
2. Tenure (strong negative correlation)
3. IsActiveMember (negative)
4. Balance (negative)
5. NumOfProducts (negative)

---

## 🔍 Model Performance Breakdown

### Generalization Assessment
✅ **NO OVERFITTING DETECTED**
- Train/Test AUC gap: -0.17% (negligible)
- Recall improves on test data: +6.43%
- F1-Score stable across folds
- Learning curves show healthy convergence

### By Class Performance
**Retained Customers (0):**
- True Negatives: 23,164
- Predicted correctly: 92.2%

**Churned Customers (1):**
- True Positives: 4,830
- Recall: 69.16% (catches 2 in 3 churners)
- Precision: 62.82% (low false alarm rate)

---

## 📈 Outputs Generated

| File | Description |
|------|-------------|
| `churn_predictions.csv` | Full dataset with churn probabilities |
| `model_comparison.csv` | Metrics comparison (train vs test) |
| `model_comparison.png` | 4-panel comparison visualization |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curves for all 3 models |
| `precision_recall.png` | Precision-recall curve |
| `shap_importance.png` | Feature importance via SHAP |
| `ann_history.png` | ANN training history |
| `learning_curve.png` | Learning curve analysis |
| `ann_model.h5` | Trained ANN weights |
| `best_xgb_pipe.pkl` | Trained XGBoost pipeline |
| `best_threshold.pkl` | Optimal decision threshold |

---

## 🔄 Deployment (OPTIONAL)

### Azure Data Lake + Data Factory (Secondary Pipeline)
For automated daily scoring and updates:

1. **Setup (optional):**
   - Upload `churn_bank.py` to Azure Data Lake
   - Create Azure Data Factory pipeline with:
     - Copy activity: ADLS → Snowflake staging
     - SQL activity: Run transformations
     - Python activity: Score predictions
     - Copy activity: Write predictions back to CHURN_SCORES

2. **Schedule:**
   - Daily trigger at 02:00 UTC
   - Automatically retrains and updates predictions

3. **Benefits:**
   - Hands-off automated scoring
   - Daily model retraining
   - Predictions always fresh in Snowflake

### GitHub Actions (CI/CD) - Optional
Monthly retraining (see `.github/workflows/model_retraining.yml`):
- Trigger new training on latest data
- Compare metrics vs baseline
- Auto-update Power BI dataset
- Create pull request with results

---

## 💻 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Data Warehouse** | Snowflake (X-Small, auto-suspend 60s) |
| **ML Framework** | TensorFlow 2.13, XGBoost 2.0, scikit-learn 1.3 |
| **Feature Eng** | SHAP 0.42, imbalanced-learn 0.11 (SMOTE) |
| **Visualization** | Power BI, Tableau, Matplotlib, Seaborn |
| **Language** | Python 3.10, SQL |
| **Version Control** | GitHub |
| **Optional Cloud** | Azure Data Lake Storage Gen2, Azure Data Factory |

---

## 📋 Requirements

```
snowflake-connector-python==3.0.0
snowflake-sqlalchemy==1.4.7
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
tensorflow==2.13.0
shap==0.42.1
imbalanced-learn==0.11.0
lightgbm==4.0.0
matplotlib==3.7.1
seaborn==0.12.2
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🧪 Testing & Validation

```bash
# Run cross-validation
python -c "from ml.churn_bank import cross_validation_results; print(cross_validation_results())"

# Generate comparison report
python ml/model_comparison.py

# Validate predictions
python ml/snowflake_writeback.py --validate
```

---

## 📞 Support & Contact

- **Questions?** Open an issue on GitHub
- **Snowflake Account:** rkiezxo-rp23355.snowflakecomputing.com
- **Data Source:** Kaggle Bank Customer Churn Dataset

---

## 📝 License

GNU General Public License v3.0 - see LICENSE file for details

This project is licensed under GPL v3. Any derivative work must also be open-source.

---

## 🙏 Acknowledgments

- **Dataset:** Kaggle Bank Customer Churn
- **ML Libraries:** TensorFlow, XGBoost, scikit-learn, SHAP
- **Data Warehouse:** Snowflake

---

**Last Updated:** March 20, 2026  
**Model Version:** Ensemble v1.0  
**Status:** ✅ Production Ready
