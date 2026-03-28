# Power BI Dashboards

## Churn_Analytics_Dashboard.pbix

Complete Power BI dashboard connected to Snowflake CHURN_ANALYTICS database.

### Features
- KPI cards: Total customers, churn rate, model metrics
- RFM matrix: Segmentation with frequency and monetary values
- Geographic analysis: Churn by country
- Risk distribution: HIGH/MEDIUM/LOW customer tiers
- Cohort retention: Retention over time by age group
- Scatter plot: Age vs Balance colored by churn status
- Precision-recall visualization
- Model performance metrics (Accuracy, ROC-AUC, Precision, Recall, F1)

### How to Use

1. **Open in Power BI Desktop**
   - Download: https://powerbi.microsoft.com/desktop
   - File → Open → Select `Churn_Analytics_Dashboard.pbix`

2. **Connect to Snowflake**
   - Update data source connection
   - Account: rkiezxo-rp23355
   - Database: CHURN_ANALYTICS
   - Warehouse: CHURN_WH
   - Refresh to get latest data

3. **Interact with Visuals**
   - Click filters to drill down
   - Hover over charts for tooltips
   - Cross-filter between visuals

### Data Source

Connects to Snowflake tables:
- BANK_CHURN_RAW (raw customer data)
- BANK_RFM_SCORES (RFM segmentation)
- BANK_COHORT_RETENTION (cohort analysis)
- BANK_ML_FEATURES (36 engineered features)
- CHURN_SCORES (ML predictions)

### DAX Measures

Key measures used:
- Total Customers = COUNTA(BANK_ML_FEATURES[CUSTOMERID])
- Churn Rate = Total Churned / Total Customers
- Model Accuracy = 0.8481
- Model ROC-AUC = 0.8867

### Requirements

- Power BI Desktop (free) or Power BI Pro (with online publishing)
- Snowflake account access
- Network access to rkiezxo-rp23355.snowflakecomputing.com