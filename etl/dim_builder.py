#  In this .py I'll Take the feature-engineered fraud_features.csv file and turn it into SQL-ready dimension tables plus a fact table with keys resolved.
#
# Input:
#   ../data/processed/fraud_features.csv
#
# Outputs:
#   ../data/processed/DimCustomer.csv
#   ../data/processed/DimMerchant.csv
#   ../data/processed/DimDate.csv
#   ../data/processed/DimAlertType.csv
#   ../data/processed/FactTransaction.csv
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import hashlib
import os


FEATURES_PATH = '../data/processed/fraud_features.csv'
DIMS_PATH     = '../data/processed/'

#? ─────────────────────────────────────────────────────────────
#? Load the Features File
#? ─────────────────────────────────────────────────────────────
def load(path: str) -> pd.DataFrame:
    """
    Load the feature-engineered fraud dataset.

    This CSV is the output from the previous feature engineering step.
    It already contains the cleaned transaction data plus the new analytical columns such as fraud_score, risk_tier, and fraud_type.

    """

    print("Loading feature-engineered data...")

    df = pd.read_csv(
        path,
        parse_dates=['trans_date_trans_time']
    )

    # Show a quick summary so we can confirm the file loaded correctly.
    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {df.shape[1]}")

    return df

#! ─────────────────────────────────────────────────────────────
#! Build DimCustomer
#! ─────────────────────────────────────────────────────────────

def build_dim_customer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the customer dimension table. So Each unique masked credit card represents one customer.

    Tis is done because:
        1. The transaction file has many rows per card because the same customer can make many transactions.
        2. In the dimension table, I only keep one clean customer record and connect transactions to it later using customer_key.
    So the output will be DimCustomer table with one row per unique customer.

    """

    print("\nBuilding DimCustomer...")

    # These are the customer-related columns that needed to be ketp in DimCustomer.

    cols = [
        'cc_num_masked',   # Hashed card number (customer identifier)
        'age_group',       
        'gender',          
        'province_code',   
        'city_ca',         
        'postal_code'           
    ]

    # Make sure all required customer columns exist before building the dimension.
    missing_cols = [col for col in cols if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Missing required customer columns: {missing_cols}")
    
    

    # Keep only one row for each masked card number.
    # The same card can appear many times in the transaction dataset,
    # but the customer dimension should describe the customer/card once.    
    dim = (
        df[cols]
        .drop_duplicates(subset='cc_num_masked')
        .copy()
        .reset_index(drop=True)
    )

    # Add a surrogate key for the database: to give every customer a simple integer ID for joins in SQL Server.
    dim.insert(0, 'customer_key', range(1, len(dim) + 1))

    # The dataset does not include account type, so I need to assign a fixed value based on the project assumption.
    #This is useful later if I have other types of accounts such as debit, credit , or prepaid
    dim['account_type'] = 'credit'

    # Print a quick summary so we can verify the dimension was created correctly.
    print(f"  Rows     : {len(dim):,}  (unique customers)")
    print(f"  Columns  : {list(dim.columns)}")

    return dim

#! ─────────────────────────────────────────────────────────────
#! Build DimDimMerchant
#! ─────────────────────────────────────────────────────────────
def build_dim_merchant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each unique Canadian merchant name represents one merchant row.
    This is done because:
        The transaction dataset can contain the same merchant many times, and  because customers may buy from the same merchant repeatedly.
        In the merchant dimension, I only want to store each merchant once.
    """

    print("\nBuilding DimMerchant...")

    # These are the merchant-related columns I want to keep.
    # merchant_ca and city_ca/province_code came from the Canadianization step.
    # category came from the original transaction dataset.
    cols = [
        'merchant_ca',
        'category',
        'province_code',
        'city_ca'
    ]

   
    missing_cols = [col for col in cols if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Missing required merchant columns: {missing_cols}")

    # I keep one row per Canadian merchant name.
    # The same merchant can appear in many transactions, but the dimension but dim table should describe the merchant only once.
    dim = (
        df[cols]
        .drop_duplicates(subset='merchant_ca')
        .copy()
        .reset_index(drop=True)
    )

    # I add a simple integer surrogate key for SQL Server joins.
    # This will be used later as merchant_key in FactTransaction.
    dim.insert(0, 'merchant_key', range(1, len(dim) + 1))

    # These categories are treated as higher-risk based on the project logic.
    high_risk_cats = {
        'shopping_net',
        'misc_net',
        'grocery_net'
    }

    dim['is_high_risk_category'] = (
        dim['category']
        .isin(high_risk_cats)
        .astype(int)
    )

    print(f"  Rows     : {len(dim):,}  (unique merchants)")
    print(f"  High-risk: {dim['is_high_risk_category'].sum():,} merchants")
    print(f"  Columns  : {list(dim.columns)}")

    return dim

#! ─────────────────────────────────────────────────────────────
#! Build DimDate
#! ─────────────────────────────────────────────────────────────

def build_dim_date() -> pd.DataFrame:
    """
    Build a complete calendar dimension table.

    I generate this table from scratch because a date dimension should include every day in the analysis range, even days with no transactions.
    The range covers 2019-01-01 to 2020-06-21 to match the dataset.
    """

    print("\nBuilding DimDate...")
    # I create one row for each day in the dataset's date range.
    dates = pd.date_range('2019-01-01', '2020-06-21', freq='D')
    dim = pd.DataFrame({'full_date': dates})

    # Date key in YYYYMMDD format for SQL Server joins.
    dim['date_key'] = dim['full_date'].dt.strftime('%Y%m%d').astype(int)

    # Calendar fields for reporting and Power BI slicing.
    dim['year'] = dim['full_date'].dt.year
    dim['quarter'] = dim['full_date'].dt.quarter
    dim['month_num'] = dim['full_date'].dt.month
    dim['month_name'] = dim['full_date'].dt.strftime('%B')
    dim['week_of_year'] = dim['full_date'].dt.isocalendar().week.astype(int)
    dim['day_of_week'] = dim['full_date'].dt.weekday + 1
    dim['day_name'] = dim['full_date'].dt.strftime('%A')

    # Weekend flag: Saturday and Sunday are marked as 1.
    dim['is_weekend'] = (dim['day_of_week'] >= 6).astype(int)

    # I only include fixed-date Canadian holidays here.
    # Moving holidays can be added later if I need more detailed holiday analysis.
    ca_holidays = {
        (1, 1),    # New Year's Day
        (7, 1),    # Canada Day
        (11, 11),  # Remembrance Day
        (12, 25),  # Christmas Day
        (12, 26),  # Boxing Day
    }

    dim['is_holiday_ca'] = dim['full_date'].apply(
        lambda d: int((d.month, d.day) in ca_holidays)
    )
    dim = dim[
    [
        'date_key',
        'full_date',
        'year',
        'quarter',
        'month_num',
        'month_name',
        'week_of_year',
        'day_of_week',
        'day_name',
        'is_weekend',
        'is_holiday_ca'
    ]
]

    print(f"  Rows     : {len(dim):,}  (daily rows 2019-01-01 to 2020-06-21)")
    print(f"  Holidays : {dim['is_holiday_ca'].sum()} flagged")
    print(f"  Columns  : {list(dim.columns)}")

    return dim

#! ─────────────────────────────────────────────────────────────
#! Build DimAlertType
#! ─────────────────────────────────────────────────────────────
def build_dim_alert_type() -> pd.DataFrame:
    """
   I create this as a small manual lookup table because the fraud alert types are controlled values from my feature engineering step.

    The fraud_type values must match exactly with the values created in feature_engineering.py so I can resolve alert_key later in FactTransaction.
    """

    print("\nBuilding DimAlertType...")

    # I define this lookup manually because the alert types are limited and controlled by my own fraud_type feature logic.
    dim = pd.DataFrame(
        [
            [1, 'card_not_present', 'Card Not Present', 'HIGH',
             'Online transaction without physical card present'],
            [2, 'atm_skimming', 'ATM Skimming', 'CRITICAL',
             'Card data captured at physical POS or ATM terminal'],
            [3, 'identity_theft', 'Identity Theft', 'HIGH',
             'Fraudster using stolen customer identity at retail'],
            [4, 'account_takeover', 'Account Takeover', 'CRITICAL',
             'Fraudster controls entire account, multiple channels'],
            [5, 'other_fraud', 'Other Fraud', 'MEDIUM',
             'Fraud transaction that does not fit the main fraud type rules'],
        ],
        columns=[
            'alert_key',
            'fraud_type',
            'fraud_label',
            'severity',
            'description'
        ]
    )

    # I check the manual lookup to avoid duplicate keys or duplicate codes.
    if dim['alert_key'].duplicated().any():
        raise ValueError("Duplicate alert_key found in DimAlertType.")

    if dim['fraud_type'].duplicated().any():
        raise ValueError("Duplicate fraud_type found in DimAlertType.")

    print(f"  Rows     : {len(dim)} alert types")
    print(f"  Columns  : {list(dim.columns)}")

    return dim

#! ─────────────────────────────────────────────────────────────
#!  Build FactTransaction
#! ─────────────────────────────────────────────────────────────

def build_fact_transaction(
    df: pd.DataFrame,
    dim_customer: pd.DataFrame,
    dim_merchant: pd.DataFrame,
    dim_alert: pd.DataFrame
) -> pd.DataFrame:
    """
    Each row represents one transaction.
    I resolve the surrogate keys from the dimension tables so this fact table is ready for SQL Server star schema loading.
    """

    print("\nBuilding FactTransaction...")

    # I work on a copy so I do not accidentally change the original dataset.
    fact = df.copy()

    required_cols = [
        'cc_num_masked',
        'merchant_ca',
        'trans_date_trans_time',
        'fraud_type',
        'is_fraud',
        'category',
        'trans_num',
        'amount_cad',
        'fraud_score',
        'transaction_hour',
        'is_weekend',
        'amount_zscore'
    ]

    missing_cols = [col for col in required_cols if col not in fact.columns]

    if missing_cols:
        raise KeyError(f"Missing required fact columns: {missing_cols}")

    # I create lookup maps from each dimension table.
    # These maps let me replace business identifiers with surrogate keys.
    cust_map = dim_customer.set_index('cc_num_masked')['customer_key']
    merch_map = dim_merchant.set_index('merchant_ca')['merchant_key']
    alert_map = dim_alert.set_index('fraud_type')['alert_key']

    fact['customer_key'] = fact['cc_num_masked'].map(cust_map)
    fact['merchant_key'] = fact['merchant_ca'].map(merch_map)

    # The date_key must match the YYYYMMDD format used in DimDate.
    fact['date_key'] = (
        fact['trans_date_trans_time']
        .dt.strftime('%Y%m%d')
        .astype(int)
    )

    # alert_key only applies to fraud rows.
    # Legitimate rows stay NULL because they do not have a fraud alert type.
    fact['alert_key'] = fact['fraud_type'].map(alert_map)
    fact.loc[fact['is_fraud'] == 0, 'alert_key'] = np.nan
    fact['alert_key'] = fact['alert_key'].astype('Int64')

    # I derive transaction_type from category because the source data does not provide a direct transaction channel column.
    online_cats = {'shopping_net', 'misc_net', 'grocery_net'}
    atm_cats = {'gas_transport', 'grocery_pos', 'misc_pos'}

    def get_txn_type(cat):
        if cat in online_cats:
            return 'online'

        if cat in atm_cats:
            return 'atm'

        return 'in-store'

    fact['transaction_type'] = fact['category'].apply(get_txn_type)

    # I keep only the columns needed for the fact table.
    # Descriptive fields are stored in dimensions, not repeated here.
    fact_cols = [
        'customer_key',
        'merchant_key',
        'date_key',
        'alert_key',
        'trans_num',
        'amount_cad',
        'is_fraud',
        'fraud_score',
        'transaction_type',
        'transaction_hour',
        'is_weekend',
        'amount_zscore',
    ]

    fact = fact[fact_cols].copy()

    # I add a transaction surrogate key for a clean SQL primary key.
    fact.insert(0, 'transaction_key', range(1, len(fact) + 1))

    print(f"  Rows          : {len(fact):,}")
    print(f"  Fraud rows    : {fact['is_fraud'].sum():,}")
    print(f"  Null cust_key : {fact['customer_key'].isna().sum():,}")
    print(f"  Null merch_key: {fact['merchant_key'].isna().sum():,}")
    print(f"  Null alert_key: {fact['alert_key'].isna().sum():,}  (expected for non-fraud rows)")
    print(f"  Columns       : {list(fact.columns)}")

    return fact

#! ─────────────────────────────────────────────────────────────
#!  Validate: No Orphaned Keys
#! ─────────────────────────────────────────────────────────────

def validate_keys(
    fact: pd.DataFrame,
    dim_customer: pd.DataFrame,
    dim_merchant: pd.DataFrame,
    dim_date: pd.DataFrame
) -> None:
    """
    Validate foreign keys before loading to SQL Server.

    I check for two things:
        1. Missing/null foreign keys in FactTransaction.
        2. Orphaned keys that do not exist in the dimension tables.
    """

    print("\n=== FK Validation ===")

    cust_keys = set(dim_customer['customer_key'])
    merch_keys = set(dim_merchant['merchant_key'])
    date_keys = set(dim_date['date_key'])

    null_cust = fact['customer_key'].isna().sum()
    null_merch = fact['merchant_key'].isna().sum()
    null_date = fact['date_key'].isna().sum()

    orphan_cust = fact[
        fact['customer_key'].notna()
        & ~fact['customer_key'].isin(cust_keys)
    ]

    orphan_merch = fact[
        fact['merchant_key'].notna()
        & ~fact['merchant_key'].isin(merch_keys)
    ]

    orphan_date = fact[
        fact['date_key'].notna()
        & ~fact['date_key'].isin(date_keys)
    ]

    print(f"  Null customer_key     : {null_cust:,}")
    print(f"  Null merchant_key     : {null_merch:,}")
    print(f"  Null date_key         : {null_date:,}")

    print(f"  Orphaned customer_key : {len(orphan_cust):,}")
    print(f"  Orphaned merchant_key : {len(orphan_merch):,}")
    print(f"  Orphaned date_key     : {len(orphan_date):,}")

    if (
        null_cust == 0
        and null_merch == 0
        and null_date == 0
        and len(orphan_cust) == 0
        and len(orphan_merch) == 0
        and len(orphan_date) == 0
    ):
        print("  All required foreign keys valid, ready for SQL load")
    else:
        print("  Fix foreign key issues before loading to SQL")
#* ─────────────────────────────────────────────────────────────
#* Main
#* ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load(FEATURES_PATH)

    dim_customer  = build_dim_customer(df)
    dim_merchant  = build_dim_merchant(df)
    dim_date      = build_dim_date()
    dim_alert     = build_dim_alert_type()
    fact          = build_fact_transaction(df, dim_customer, dim_merchant, dim_alert)

    validate_keys(fact, dim_customer, dim_merchant, dim_date)

    os.makedirs(DIMS_PATH, exist_ok=True)
    dim_customer.to_csv(DIMS_PATH + 'DimCustomer.csv',     index=False)
    dim_merchant.to_csv(DIMS_PATH + 'DimMerchant.csv',     index=False)
    dim_date.to_csv    (DIMS_PATH + 'DimDate.csv',         index=False)
    dim_alert.to_csv   (DIMS_PATH + 'DimAlertType.csv',    index=False)
    fact.to_csv        (DIMS_PATH + 'FactTransaction.csv',  index=False)


