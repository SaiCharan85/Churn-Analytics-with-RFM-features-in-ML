-- ============================================================
-- SET CONTEXT
-- ============================================================
USE WAREHOUSE CHURN_WH;
USE DATABASE CHURN_ANALYTICS;
USE SCHEMA ANALYTICS;

-- ============================================================
-- 1. BANK RFM SCORES TABLE
-- ============================================================
CREATE OR REPLACE TABLE CHURN_ANALYTICS.ANALYTICS.BANK_RFM_SCORES (
    CUSTOMER_ID         VARCHAR,
    CUSTOMERID          INTEGER,
    RECENCY             NUMBER,
    FREQUENCY           FLOAT,
    MONETARY            FLOAT,
    BALANCE             FLOAT,
    ESTIMATEDSALARY     FLOAT,
    CREDITSCORE         INTEGER,
    AGE                 FLOAT,
    TENURE              INTEGER,
    GEOGRAPHY           VARCHAR,
    GENDER              VARCHAR,
    HASCRCARD           FLOAT,
    NUMOFPRODUCTS       INTEGER,
    ISACTIVEMEMBER      FLOAT,
    EXITED              INTEGER,
    R_SCORE             NUMBER,
    F_SCORE             NUMBER,
    M_SCORE             NUMBER,
    RFM_TOTAL           NUMBER,
    RFM_SEGMENT         VARCHAR
);

-- ============================================================
-- 2. BANK COHORT RETENTION TABLE
-- ============================================================
CREATE OR REPLACE TABLE CHURN_ANALYTICS.ANALYTICS.BANK_COHORT_RETENTION (
    GEOGRAPHY           VARCHAR,
    AGE_GROUP           VARCHAR,
    COHORT_SIZE         NUMBER,
    CHURNED_COUNT       NUMBER,
    CHURN_RATE_PCT      FLOAT,
    AVG_TENURE          FLOAT,
    RETENTION_0_2YR     FLOAT,
    RETENTION_0_5YR     FLOAT,
    RETENTION_0_10YR    FLOAT
);

-- ============================================================
-- 3. BANK ML FEATURES TABLE
-- ============================================================
CREATE OR REPLACE TABLE CHURN_ANALYTICS.ANALYTICS.BANK_ML_FEATURES (
    CUSTOMER_ID             VARCHAR,
    CUSTOMERID              INTEGER,
    AGE                     FLOAT,
    GENDER                  VARCHAR,
    GEOGRAPHY               VARCHAR,
    CREDITSCORE             INTEGER,
    TENURE                  INTEGER,
    BALANCE                 FLOAT,
    NUMOFPRODUCTS           INTEGER,
    HASCRCARD               FLOAT,
    ISACTIVEMEMBER          FLOAT,
    ESTIMATEDSALARY         FLOAT,
    R_SCORE                 NUMBER,
    F_SCORE                 NUMBER,
    M_SCORE                 NUMBER,
    RFM_TOTAL               NUMBER,
    RFM_SEGMENT             VARCHAR,
    RECENCY                 NUMBER,
    FREQUENCY               FLOAT,
    MONETARY                FLOAT,
    BALANCE_SALARY_RATIO    FLOAT,
    CREDIT_AGE_RATIO        FLOAT,
    BALANCE_PER_PRODUCT     FLOAT,
    TENURE_AGE_RATIO        FLOAT,
    ACTIVE_X_PRODUCTS       FLOAT,
    CREDIT_X_ACTIVE         FLOAT,
    BALANCE_X_ACTIVE        FLOAT,
    HAS_BALANCE             NUMBER,
    SENIOR_CUSTOMER         NUMBER,
    LONG_TENURE             NUMBER,
    RFM_X_CREDIT            FLOAT,
    RFM_X_BALANCE           FLOAT,
    COHORT_CHURN_RATE       FLOAT,
    COHORT_AVG_TENURE       FLOAT,
    COHORT_SIZE             NUMBER,
    IS_CHURNED              INTEGER
);

-- ============================================================
-- 4. CHURN SCORES TABLE (ML output writeback)
-- ============================================================
CREATE OR REPLACE TABLE CHURN_ANALYTICS.ANALYTICS.CHURN_SCORES (
    CUSTOMER_ID         VARCHAR,
    CHURN_PROBABILITY   FLOAT,
    CHURN_LABEL         BOOLEAN,
    RISK_TIER           VARCHAR,
    MODEL_VERSION       VARCHAR,
    SCORED_AT           TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ============================================================
-- VERIFY ALL TABLES CREATED
-- ============================================================
SHOW TABLES IN SCHEMA CHURN_ANALYTICS.ANALYTICS;