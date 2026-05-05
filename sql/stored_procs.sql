USE FraudDB;
GO

-- ============================================================
-- Stored Procedure: 
--
-- 
-- This procedure gives a province-level fraud summary for any selected date range.
--
-- I use this procedure instead of repeating the same query every time I want to analyze fraud by province.
--
-- Parameters:
--   @StartDate: first date included in the analysis
--   @EndDate  : last date included in the analysis
--
-- Notes:
--   is_fraud is a BIT column in SQL Server, so I cast it to INT before using SUM().
-- ============================================================

CREATE PROCEDURE dbo.usp_GetFraudSummaryByProvince -- usp: user stored procedure
    @StartDate DATE,
    @EndDate   DATE
AS -- This means that the next step is the code that will be executed inside the stored procedure.
BEGIN -- The beginning of procedure body (from BEGIN To END)

    -- Prevents SQL Server from returning extra row count messages.
    -- This keeps the procedure output cleaner, especially when used later by Power BI.
    SET NOCOUNT ON;

    SELECT
        
        c.province_code, -- Province comes from DimCustomer because each transaction is linked to a customer through customer_key.
        COUNT(*) AS total_transactions,-- Total number of transactions in this province within the selected date range.
        SUM(CAST(f.is_fraud AS INT)) AS fraud_count, -- is_fraud is stored as BIT, so it must be converted to INT before using SUM().

        -- Total fraud amount in CAD.
        -- Only fraud transactions are included in the loss amount.
        SUM(
            CASE
                WHEN f.is_fraud = 1 THEN f.amount_cad
                ELSE 0
            END
        ) AS fraud_loss_cad,

        -- Average fraud score for fraud transactions only.
        -- Non-fraud rows return NULL here, and AVG ignores NULL values.
        AVG(
            CASE
                WHEN f.is_fraud = 1 THEN f.fraud_score
                ELSE NULL
            END
        ) AS avg_fraud_score,

        -- Fraud rate as a percentage:
        -- fraud transactions / total transactions * 100
        -- 100.0 is used to force decimal calculation.
        CAST(
            SUM(CAST(f.is_fraud AS INT)) * 100.0 / COUNT(*)
            AS DECIMAL(5,2)
        ) AS fraud_rate_pct

    FROM dbo.FactTransaction AS f

    -- Join to customer dimension to get province_code.
    INNER JOIN dbo.DimCustomer AS c
        ON f.customer_key = c.customer_key

    -- Join to date dimension so we can filter by the real date.
    INNER JOIN dbo.DimDate AS d
        ON f.date_key = d.date_key

    -- Date range selected by the user when running the procedure.
    WHERE d.full_date BETWEEN @StartDate AND @EndDate

    -- One summary row per province.
    GROUP BY
        c.province_code

    -- Show the provinces with the most fraud first.
    ORDER BY
        fraud_count DESC;
END;
GO
-- ===========Test ====================
EXEC dbo.usp_GetFraudSummaryByProvince
    @StartDate = '2019-01-01',
    @EndDate   = '2020-06-21';
-- ====================================
