-- =================================================================
-- Query 1: Top Fraud Provinces
--
-- This query summarizes fraud activity by customer province.
-- It helps identify which provinces have the highest fraud volume,fraud loss, and fraud rate.
--
-- Why this matters:
-- A province may have a high fraud count because it has many total transactions,
-- while another province may have a smaller fraud count but a higher fraud rate.
-- Both views are useful for fraud analysis.
--
-- Tables used:
--   FactTransaction: transaction-level facts such as amount, fraud flag
--   DimCustomer    : customer details, including province_code
--
-- ==================================================================

SELECT
    c.province_code,-- Province where the customer is located.
    COUNT(*) AS total_transactions, -- Total number of transactions in each province.
    SUM(CAST(f.is_fraud AS INT)) AS fraud_count,
    -- Total CAD amount lost to fraud. Only fraud transactions are included in this calculation.
    SUM(
        CASE
            WHEN f.is_fraud = 1 THEN f.amount_cad
            ELSE 0
        END
    ) AS fraud_loss_cad,

    CAST(
        SUM(CAST(f.is_fraud AS INT)) * 100.0 / COUNT(*)
        AS DECIMAL(5,2)
    ) AS fraud_rate_pct

FROM dbo.FactTransaction AS f

-- Join to DimCustomer to get the province for each transaction.
INNER JOIN dbo.DimCustomer AS c
    ON f.customer_key = c.customer_key

-- One result row per province.
GROUP BY
    c.province_code

-- Provinces with the highest fraud count appear first.
ORDER BY
    fraud_count DESC;
GO

-- ====================================================================================
-- Query 2: Peak Fraud Hours
--
-- 
-- This query analyzes fraud activity by transaction hour.
-- It helps identify which hours of the day have the highest fraud volume and fraud rate.
--
-- Why this matters:
-- Fraud may not happen evenly throughout the day. 
-- Some hours may have more fraud transactions or a higher fraud rate, which can help guide monitoring rules and dashboard insights.
--
-- ======================================================================================

SELECT
    f.transaction_hour, -- Hour of the day when the transaction happened.    
    COUNT(*) AS total_transactions,-- Total number of transactions during this hour.
    SUM(CAST(f.is_fraud AS INT)) AS fraud_count,-- Total number of fraud transactions during this hour.

    -- Fraud rate within this hour.
    -- Example: fraud transactions at 2 AM / all transactions at 2 AM.
    CAST(
        SUM(CAST(f.is_fraud AS INT)) * 100.0 / COUNT(*)
        AS DECIMAL(5,2)
    ) AS fraud_rate_pct,

    -- Share of total fraud represented by this hour.
    -- Example: fraud at 2 AM / all fraud transactions across all hours.
    CAST(
        SUM(CAST(f.is_fraud AS INT)) * 100.0
        / SUM(SUM(CAST(f.is_fraud AS INT))) OVER ()
        AS DECIMAL(5,2)
    ) AS pct_of_all_fraud

FROM dbo.FactTransaction AS f

-- One summary row per transaction hour.
GROUP BY
    f.transaction_hour

-- I want to see hours from midnight to 11 PM.
ORDER BY
    f.transaction_hour;
GO


-- ====================================================================================
-- Query 3: Fraud by Merchant Category
--
-- 
-- This query summarizes fraud activity by merchant category.
-- It helps identify which transaction categories have the highest fraud rates, fraud counts, and fraud losses.
--
-- Why this matters:
-- Some categories may naturally have more fraud exposure than others.
-- For example, online shopping or miscellaneous online transactions may show higher fraud risk than routine in-person categories.
--
-- Tables used:
--   FactTransaction: transaction-level facts such as amount, fraud flag, fraud score, and merchant_key
--   DimMerchant    : merchant details, including transaction category
--
-- ====================================================================================

SELECT
    
    m.category,
    COUNT(*) AS total_transactions,
    SUM(CAST(f.is_fraud AS INT)) AS fraud_count,
    SUM(
        CASE
            WHEN f.is_fraud = 1 THEN f.amount_cad
            ELSE 0
        END
    ) AS fraud_loss_cad,

    -- Fraud rate within each category:
    CAST(
        SUM(CAST(f.is_fraud AS INT)) * 100.0 / COUNT(*)
        AS DECIMAL(5,2)
    ) AS fraud_rate_pct,

    -- Average fraud score for fraudulent transactions only.
    CAST(
        AVG(
            CASE
                WHEN f.is_fraud = 1 THEN f.fraud_score
                ELSE NULL
            END
        )
        AS DECIMAL(5,2)
    ) AS avg_fraud_score

FROM dbo.FactTransaction AS f

-- Join to DimMerchant to bring in merchant category.
INNER JOIN dbo.DimMerchant AS m
    ON f.merchant_key = m.merchant_key

-- One summary row per merchant category.
GROUP BY
    m.category

-- Highest fraud-rate categories appear first.
ORDER BY
    fraud_rate_pct DESC;
GO


-- ====================================================================================
-- Query 4: High Risk Customers
--
-- 
-- This query identifies the top 20 customers with the highest number of fraud transactions.
--
-- Why this matters:
-- Some customers may appear more often in fraud cases than others.
-- This query helps identify customer profiles that may need closer monitoring, such as repeated fraud activity by age group or province.
--
-- Important:
-- cc_num_masked is used instead of the raw card number to protect sensitive customer information.
-- ====================================================================================
SELECT TOP 20
    c.cc_num_masked,
    c.age_group,
    c.province_code,
    COUNT(*) AS total_transactions,
    SUM(CAST(f.is_fraud AS INT)) AS fraud_count,
    SUM(
        CASE
            WHEN f.is_fraud = 1 THEN f.amount_cad
            ELSE 0
        END
    ) AS total_fraud_loss_cad,
    CAST(
        AVG(
            CASE
                WHEN f.is_fraud = 1 THEN f.fraud_score
                ELSE NULL
            END
        )
        AS DECIMAL(5,2)
    ) AS avg_fraud_score

FROM dbo.FactTransaction AS f

-- Join to DimCustomer to bring in masked customer details.
INNER JOIN dbo.DimCustomer AS c
    ON f.customer_key = c.customer_key

-- One summary row per customer profile.
GROUP BY
    c.cc_num_masked,
    c.age_group,
    c.province_code

-- Keep only customers with at least one fraud transaction.
HAVING
    SUM(CAST(f.is_fraud AS INT)) > 0

-- Customers with the highest fraud count appear first.
ORDER BY
    fraud_count DESC,
    total_fraud_loss_cad DESC;
GO


-- ====================================================================================
-- Query 5: Monthly Fraud Trend
--
-- Purpose:
-- This query summarizes fraud activity by year and month.
-- It helps show whether fraud volume, fraud loss, and fraud rate increase or decrease over time.
--
-- Why this matters:
-- Monthly trend analysis is useful for Power BI line charts and time-based reporting. 
-- It helps identify seasonal fraud patterns, sudden increases, or months with unusually high fraud losses.
--
--
-- ====================================================================================

SELECT
    
    d.year,
    d.month_num,
    d.month_name,
    COUNT(*) AS total_transactions,-- Total number of transactions in each month.
    SUM(CAST(f.is_fraud AS INT)) AS fraud_count, -- Total number of fraud transactions in each month.
    SUM(
        CASE
            WHEN f.is_fraud = 1 THEN f.amount_cad
            ELSE 0
        END
    ) AS fraud_loss_cad,

   CAST(
        SUM(CAST(f.is_fraud AS INT)) * 100.0 / COUNT(*)
        AS DECIMAL(5,2)
    ) AS fraud_rate_pct

FROM dbo.FactTransaction AS f

-- Join to DimDate so transactions can be grouped by calendar month.
INNER JOIN dbo.DimDate AS d
    ON f.date_key = d.date_key

-- One summary row per year and month.
GROUP BY
    d.year,
    d.month_num,
    d.month_name

-- Sort results in proper time order.
ORDER BY
    d.year,
    d.month_num;
GO