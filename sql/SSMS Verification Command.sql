USE FraudDB;

SELECT COUNT(*)                            AS total_transactions,
       SUM(CAST(is_fraud AS INT))          AS fraud_count,
       CAST(
           SUM(CAST(is_fraud AS INT)) * 100.0 / COUNT(*)
       AS DECIMAL(5,2))                    AS fraud_pct
FROM dbo.FactTransaction;

SELECT TOP 20
    f.trans_num,
    c.age_group,
    c.province_code AS customer_province,
    m.merchant_ca,
    m.category,
    d.full_date,
    a.fraud_label,
    f.amount_cad,
    f.is_fraud,
    f.fraud_score
FROM dbo.FactTransaction f
JOIN dbo.DimCustomer c
    ON f.customer_key = c.customer_key
JOIN dbo.DimMerchant m
    ON f.merchant_key = m.merchant_key
JOIN dbo.DimDate d
    ON f.date_key = d.date_key
LEFT JOIN dbo.DimAlertType a
    ON f.alert_key = a.alert_key;