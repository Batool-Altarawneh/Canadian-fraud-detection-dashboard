-- ============================================================
-- DimCustomer
-- Stores customer-related information used for analysis.
--
-- I keep customer details in a separate dimension table so the fact table does not repeat the same customer information for every transaction.
--
-- cc_num_masked is unique because each masked card number should identify one customer record in this project.
-- ============================================================

CREATE TABLE DimCustomer (
    customer_key  INT          NOT NULL PRIMARY KEY,
    cc_num_masked VARCHAR(64)  NOT NULL UNIQUE,
    age_group     VARCHAR(10)  NOT NULL,
    gender        CHAR(1)      NOT NULL,
    province_code CHAR(2)      NOT NULL,
    city_ca       VARCHAR(60)  NOT NULL,
    postal_code   VARCHAR(7)   NOT NULL,
    account_type  VARCHAR(20)  NOT NULL
);
GO

-- ============================================================
-- DimMerchant
-- Stores merchant information such as merchant name, category,and Canadian location details.
--
-- This table helps analyze fraud patterns by merchant category,city, province, and high-risk merchant type.
--
-- merchant_ca is unique because each cleaned Canadian merchant name should appear only once in this dimension.
-- ============================================================

CREATE TABLE DimMerchant (
    merchant_key          INT          NOT NULL PRIMARY KEY,
    merchant_ca           VARCHAR(100) NOT NULL UNIQUE,
    category              VARCHAR(40)  NOT NULL,
    province_code         CHAR(2)      NOT NULL,
    city_ca               VARCHAR(60)  NOT NULL,
    is_high_risk_category BIT          NOT NULL DEFAULT 0
);
GO

-- ============================================================
-- DimDate
-- Stores calendar attributes for each transaction date.
--
-- I separate date fields into a date dimension because it makes Power BI analysis easier for year, quarter, month, weekday,weekend, and holiday reporting.
--
-- date_key is stored as an integer like YYYYMMDD.
-- Example: 20260504 means May 4, 2026.
-- ============================================================

CREATE TABLE DimDate (
    date_key      INT         NOT NULL PRIMARY KEY,
    full_date     DATE        NOT NULL,
    year          SMALLINT    NOT NULL,
    quarter       TINYINT     NOT NULL,
    month_num     TINYINT     NOT NULL,
    month_name    VARCHAR(10) NOT NULL,
    week_of_year  TINYINT     NOT NULL,
    day_of_week   TINYINT     NOT NULL,
    day_name      VARCHAR(10) NOT NULL,
    is_weekend    BIT         NOT NULL DEFAULT 0,
    is_holiday_ca BIT         NOT NULL DEFAULT 0
);
GO
-- ============================================================
-- DimAlertType
-- Stores the controlled list of fraud alert types.
--
-- These fraud_type values must match the values generated in feature_engineering.py.
-- This is important because the ETL/load step will use fraud_type to resolve alert_key for fraud records.
--
-- Legitimate transactions do not need an alert type, so alert_key will be NULL in the fact table when is_fraud = 0.
-- ============================================================

CREATE TABLE DimAlertType (
    alert_key   INT          NOT NULL PRIMARY KEY,
    fraud_type  VARCHAR(30)  NOT NULL UNIQUE,
    fraud_label VARCHAR(50)  NOT NULL,
    severity    VARCHAR(10)  NOT NULL,
    description VARCHAR(200) NOT NULL
);
GO

-- ============================================================
-- FactTransaction
-- Stores one row per transaction.
--
-- This is the central fact table of the star schema. 
-- It connects each transaction to the customer, merchant, date, and optional fraud alert type using foreign keys.
--
-- I create this table last because it depends on the dimension tables above. 
--
-- alert_key is nullable because normal transactions do not have a fraud alert type.
--
-- trans_num is unique because it represents the original unique transaction number from the source data and helps prevent duplicate transaction records.
-- ============================================================

CREATE TABLE FactTransaction (
    transaction_key  INT           NOT NULL PRIMARY KEY,
    customer_key     INT           NOT NULL,
    merchant_key     INT           NOT NULL,
    date_key         INT           NOT NULL,
    alert_key        INT           NULL,
    trans_num        VARCHAR(50)   NOT NULL UNIQUE,
    amount_cad       DECIMAL(10,2) NOT NULL,
    is_fraud         BIT           NOT NULL,
    fraud_score      DECIMAL(5,2)  NOT NULL,
    transaction_type VARCHAR(20)   NOT NULL,
    transaction_hour TINYINT       NOT NULL,
    is_weekend       BIT           NOT NULL,
    amount_zscore    DECIMAL(8,4)  NOT NULL,
    created_at       DATETIME      NOT NULL DEFAULT GETDATE(),

    CONSTRAINT FK_Fact_Customer FOREIGN KEY (customer_key)
        REFERENCES DimCustomer (customer_key),

    CONSTRAINT FK_Fact_Merchant FOREIGN KEY (merchant_key)
        REFERENCES DimMerchant (merchant_key),

    CONSTRAINT FK_Fact_Date FOREIGN KEY (date_key)
        REFERENCES DimDate (date_key),

    CONSTRAINT FK_Fact_Alert FOREIGN KEY (alert_key)
        REFERENCES DimAlertType (alert_key)
);
GO