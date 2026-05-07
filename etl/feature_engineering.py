import pandas as pd
import numpy as np
from scipy import stats
import os

PROCESSED_PATH = '../data/processed/fraud_canadianized.csv'
FEATURES_PATH  = '../data/processed/fraud_features.csv'

#! ─────────────────────────────────────────────────────────────
#! LOAD
#! ─────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    """Load the processed CSV produced by etl_pipeline.py."""
    print("Loading processed data...")
    df = pd.read_csv(path, parse_dates=['trans_date_trans_time'])
    print(f"  Rows     : {len(df):,}")
    print(f"  Columns  : {df.shape[1]}")
    print(f"  Nulls    : {df.isnull().sum().sum()}")
    return df

#! ─────────────────────────────────────────────────────────────
#! FEATURE 1 — AGE GROUP
#! ─────────────────────────────────────────────────────────────

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a customer age group feature from the dob column.

    I use a fixed reference date instead of today's date because this dataset
    ends in 2020. This keeps the age calculation stable every time the script
    is re-run.

    Final age groups:
        18-25
        26-35
        36-50
        51-65
        65+
    """
    print("\nFeature 1: age_group")

    # Use the dataset end date so the results stay the same in the future.
    reference_date = pd.Timestamp('2020-12-31')

    # Convert dob to a datetime column.
    # errors='coerce' prevents the script from breaking if one bad date appears.
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    # Calculate age in years.
    # I divide by 365.25 to account for leap years in a simple way.
    df['age'] = ((reference_date - df['dob']).dt.days / 365.25).astype(int)

    # Basic checks to make sure the age calculation looks reasonable.
    print(f"  Age range          : {df['age'].min()} – {df['age'].max()} years")
    print(f"  Age mean           : {df['age'].mean():.1f} years")

    # Check for unrealistic ages. Since this is synthetic data, small issues can happen, so I log them instead of stopping the whole pipeline.
    under_18 = (df['age'] < 18).sum()
    over_100 = (df['age'] > 100).sum()

    if under_18 > 0:
        print(
            f"  NOTE: {under_18:,} transactions have age < 18. "
            f"They will fall into the youngest reporting bucket."
        )

    if over_100 > 0:
        print(
            f"  WARNING: {over_100:,} transactions have age > 100. "
            f"Check the dob column if this was real production data."
        )

    # right=True means the upper number is included in each group.
    # Example: age 25 goes into 18-25, age 35 goes into 26-35.
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 50, 65, 120],
        labels=['18-25', '26-35', '36-50', '51-65', '65+'],
        right=True
    ).astype(str)

    # Show how many transactions are in each age group.
    print("  Distribution       :")
    for group_name, count in df['age_group'].value_counts().sort_index().items():
        percentage = count / len(df) * 100
        print(f"    {group_name:6s} → {count:>10,}  ({percentage:.1f}%)")

    # Quick fraud summary by age group.
    # This is useful to validate that the new feature can support dashboard analysis.
    print("\n  Fraud rate by age group:")

    fraud_by_age = (
        df.groupby('age_group')['is_fraud']
          .agg(['sum', 'count'])
          .assign(fraud_rate=lambda x: (x['sum'] / x['count'] * 100).round(3))
          .rename(columns={
              'sum': 'fraud_count',
              'count': 'total'
          })
    )

    print(fraud_by_age.to_string())

    # Keep only the dashboard-friendly column.
    # dob is sensitive, and raw age is less useful than age_group for reporting.
    df = df.drop(columns=['dob', 'age'], errors='ignore')

    print("\n  Dropped            : dob, age")
    print("  Added              : age_group")

    return df


#! ─────────────────────────────────────────────────────────────
#! FEATURE 2 — AMOUNT Z-SCORE PER CUSTOMER
#! ─────────────────────────────────────────────────────────────

def add_amount_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate how unusual each transaction amount is compared to the customer's own spending history.

    This is stronger than a simple "large amount" flag because the same amount can mean different things for different customers.

    Example:
        - $500 is suspicious for a customer who usually spends $20
        - $500 is normal for a customer who usually spends $800

    The z-score tells us how far a transaction is from that customer's average transaction amount.
    """
    print("\nFeature 2: amount_zscore")

    # Make sure the columns needed for this feature exist before running.
    required_cols = ['cc_num_masked', 'amount_cad', 'is_fraud']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()

    #? ── Calculate per-customer z-score ─────────────────────────
    # cc_num_masked identifies the customer/card after removing raw Personally Identifiable Information.
    
    # transform() keeps the result aligned with the original DataFrame, so every transaction gets its own z-score.
    #
    # ddof=1 means sample standard deviation. This is reasonable here  because each customer has only a sample of transactions in the dataset.
    df['amount_zscore'] = (
        df.groupby('cc_num_masked')['amount_cad']
          .transform(lambda x: stats.zscore(x, ddof=1))  # z-score = (transaction amount - customer average amount) / customer standard deviation
    )

    #? ── Handle missing z-scores ────────────────────────────────
    # NaN can happen when:
    #   1. A customer has only one transaction
    #   2. A customer has repeated transactions with the exact same amount, causing the standard deviation to be 0
    
    # In both cases, we do not have enough variation to measure unusualness, so using 0 is a safe neutral value.
    nan_count = df['amount_zscore'].isna().sum()

    df['amount_zscore'] = (
        df['amount_zscore']
          .fillna(0)
          .round(4)
    )

    print(
        f"  NaN z-scores       : {nan_count:,} filled with 0 "
        f"(single transaction or no amount variation)"
    )

    #? ── Clip extreme values ────────────────────────────────────
    # In statistics, values beyond ±3 are usually considered extreme.
    #
    # Clipping keeps the information that the transaction is unusual, but stops one very large outlier from dominating the fraud_score later.
    
    # Meaning:
    #   z = 0: normal for this customer
    #   z = 1: somewhat higher than usual
    #   z = 2: clearly higher than usual
    #   z = 3: extremely higher than usual
    df['amount_zscore'] = df['amount_zscore'].clip(-3, 3)

    #? ── Validate the feature ───────────────────────────────────
    # If this feature is useful, fraud transactions should usually have a higher average z-score than legitimate transactions.
    legit_mean = df.loc[df['is_fraud'] == 0, 'amount_zscore'].mean()
    fraud_mean = df.loc[df['is_fraud'] == 1, 'amount_zscore'].mean()

#The median is useful because it is less affected by extreme values.
#This means that if there are a few very unusual transactions, the average might be affected, but the median gives us a better picture.
    legit_median = df.loc[df['is_fraud'] == 0, 'amount_zscore'].median()
    fraud_median = df.loc[df['is_fraud'] == 1, 'amount_zscore'].median()

    difference = fraud_mean - legit_mean

    print(f"  Legitimate txns    : mean={legit_mean:.3f}  median={legit_median:.3f}")
    print(f"  Fraud txns         : mean={fraud_mean:.3f}  median={fraud_median:.3f}")
    print(
        f"  Mean difference    : {difference:.3f} "
        f"(fraud mean minus legitimate mean)"
    )

    #? ── Show z-score distribution ──────────────────────────────
    # This gives a quick overview of how many transactions are normal, slightly unusual, or very unusual.
    print("\n  Z-score distribution:")

    bins = [-3.01, -1, 0, 1, 2, 3.01]
    labels = ['< -1', '-1 to 0', '0 to 1', '1 to 2', '2 to 3']

    zscore_bins = pd.cut(
        df['amount_zscore'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    for label, count in zscore_bins.value_counts().sort_index().items():
        percentage = count / len(df) * 100
        print(f"    {label:10s} → {count:>10,}  ({percentage:.1f}%)")

    print("\n  Added              : amount_zscore")

    return df

#! ─────────────────────────────────────────────────────────────
#! FEATURE 3 — FRAUD TYPE
#! ─────────────────────────────────────────────────────────────

def add_fraud_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a descriptive fraud type label from the transaction category.

    The original dataset only tells us whether a transaction is fraud or notusing the is_fraud column. 

    This mapping is a project business rule Im creating for the project:
        - online categories are treated as card_not_present
        - physical terminal categories are treated as atm_skimming
        - retail and dining categories are treated as identity_theft
        - larger or account-based categories are treated as account_takeover

    Legitimate transactions are assigned fraud_type = 'none'.
    """
    print("\nFeature 3: fraud_type")

    # Make sure the columns needed for this feature exist.
    required_cols = ['category', 'is_fraud']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()

    #? ── Category to fraud type mapping ──────────────────────────
    # This is not directly provided by the dataset.
    
    fraud_type_map = {
        # Online transactions where the physical card is not required.
        'shopping_net'   : 'card_not_present',
        'misc_net'       : 'card_not_present',
        'grocery_net'    : 'card_not_present',

        # Physical terminal categories where card data could be skimmed.
        'grocery_pos'    : 'atm_skimming',
        'gas_transport'  : 'atm_skimming',
        'misc_pos'       : 'atm_skimming',

        # Retail, dining, and lifestyle categories that fit an identity misuse story.
        'shopping_pos'   : 'identity_theft',
        'food_dining'    : 'identity_theft',
        'entertainment'  : 'identity_theft',
        'kids_pets'      : 'identity_theft',

        # Larger, recurring, or account-driven purchases.
        'home'           : 'account_takeover',
        'travel'         : 'account_takeover',
        'health_fitness' : 'account_takeover',
        'personal_care'  : 'account_takeover',
    }

    #? ── Check for categories not included in the mapping ─────────
    # This protects the pipeline if the dataset changes later.
    all_categories = set(df['category'].dropna().unique())
    mapped_categories = set(fraud_type_map.keys())
    unmapped = all_categories - mapped_categories

    if unmapped:
        print(
            f"  NOTE: {len(unmapped)} unmapped categories found. "
            f"They will be labeled as 'other_fraud': {unmapped}"
        )

    # ── Apply the mapping ───────────────────────────────────────
    # First, map the transaction category to a fraud type.
    # Any unknown category becomes other_fraud instead of being forced into a specific type that may not be accurate.
    df['fraud_type'] = (
        df['category']
          .map(fraud_type_map)
          .fillna('other_fraud')
    )

    # Legitimate transactions should not receive a fraud label.
    # Only fraud transactions keep the mapped fraud type.
    df.loc[df['is_fraud'] == 0, 'fraud_type'] = 'none'

    #? ── Fraud type breakdown ────────────────────────────────────
    # This shows the distribution only for fraud transactions.
    fraud_only = df[df['is_fraud'] == 1]

    print(f"  Fraud type breakdown ({len(fraud_only):,} fraud transactions):")

    if len(fraud_only) > 0:
        for fraud_type, count in fraud_only['fraud_type'].value_counts().items():
            percentage = count / len(fraud_only) * 100
            print(f"    {fraud_type:20s} → {count:>6,}  ({percentage:.1f}%)")
    else:
        print("    No fraud transactions found.")

    #? ── Validate the logic ──────────────────────────────────────
    # This confirms that legitimate transactions were not accidentally assigned a fraud type.
    legit_with_type = df[
        (df['is_fraud'] == 0) & (df['fraud_type'] != 'none')
    ].shape[0]

    if legit_with_type > 0:
        print(
            f"  WARNING: {legit_with_type:,} legitimate transactions "
            f"have a non-none fraud_type. Check the mapping logic."
        )
    else:
        print("  Validation         : all legitimate transactions = 'none' ✓")

    #? ── Full dataset distribution ───────────────────────────────
    # This includes legitimate transactions, so (none) should be the largest group.
    print("\n  Full dataset distribution:")

    for fraud_type, count in df['fraud_type'].value_counts().items():
        percentage = count / len(df) * 100
        print(f"    {fraud_type:20s} → {count:>10,}  ({percentage:.1f}%)")

    print("\n  Added              : fraud_type")

    return df


#! ─────────────────────────────────────────────────────────────
#! FEATURE 4 — FRAUD SCORE
#! ─────────────────────────────────────────────────────────────

def add_fraud_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 0-100 fraud risk score using five analytical signals.

    The goal is to turn several fraud indicators into one clear number.

    Signals used:
        amount_zscore : how unusual the amount is for the customer
        time_of_day : late-night transactions are treated as riskier
        dist_km : larger customer-to-merchant distance adds risk
        city_tier : larger cities receive a slightly higher risk weight
        is_weekend : weekend transactions receive a small risk uplift

    The weights add up to 100, so the final score is easy to explain.
    The weights here aren't pre-programmed in the data. It's a business-rule scoring model that I speciffically designe for the project.
    """
    print("\nFeature 4: fraud_score")

    # Make sure all columns needed for this score exist before running.
    required_cols = [
        'amount_zscore',
        'time_of_day',
        'dist_km',
        'city_tier',
        'is_weekend',
        'is_fraud'
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()

    #? ── Risk lookup tables ──────────────────────────────────────
    # These mappings are business rules for the project.
    # They convert readable categories into numeric risk values from 0 to 1.

    time_risk = {
        'Late Night': 1.0,   # highest risk period
        'Evening': 0.5,      # moderate risk
        'Morning': 0.2,      # lower risk
        'Afternoon': 0.1     # lowest risk
    }

    city_tier_risk = {
        'Metro': 1.0,
        'Large City': 0.7,
        'Mid City': 0.4,
        'Small Town': 0.2
    }

    #? ── Component 1: amount_zscore — 35 points ─────────────────
    # Negative z-scores mean the transaction amount is lower than usual, so they do not add risk here.
    
    # Values are clipped between 0 and 3, then scaled to 35 points.
    amount_component = (
        df['amount_zscore']
          .clip(lower=0, upper=3)
          .div(3)
          .mul(35)
    )

    print(
        f"  Component 1 (amount_zscore) : "
        f"min={amount_component.min():.2f}  "
        f"max={amount_component.max():.2f}  "
        f"mean={amount_component.mean():.2f}"
    )

    #? ── Component 2: time_of_day — 25 points ───────────────────
    # Map each time period to a risk value, then scale it by 25.
    time_component = (
        df['time_of_day']
          .map(time_risk)
          .fillna(0)
          .mul(25)
    )

    print(
        f"  Component 2 (time_of_day)   : "
        f"min={time_component.min():.2f}  "
        f"max={time_component.max():.2f}  "
        f"mean={time_component.mean():.2f}"
    )

    #? ── Component 3: distance — 20 points ──────────────────────
    # I use the 99th percentile instead of the maximum distance.
    # This keeps extreme outliers from compressing the rest of the values.
    dist_99 = df['dist_km'].quantile(0.99)

    if pd.isna(dist_99) or dist_99 == 0:
        dist_component = pd.Series(0, index=df.index)
        print("  Component 3 (dist_km)       : skipped because distance baseline is 0")
    else:
        dist_component = (
            df['dist_km']
              .clip(lower=0, upper=dist_99)
              .div(dist_99)
              .mul(20)
        )

        print(
            f"  Component 3 (dist_km)       : "
            f"99th pct={dist_99:.1f} km  "
            f"min={dist_component.min():.2f}  "
            f"max={dist_component.max():.2f}  "
            f"mean={dist_component.mean():.2f}"
        )

    #? ── Component 4: city_tier — 10 points ─────────────────────
    # Unknown city tiers get the lowest default risk instead of breaking the code.
    city_component = (
        df['city_tier']
          .map(city_tier_risk)
          .fillna(0.2)
          .mul(10)
    )

    print(
        f"  Component 4 (city_tier)     : "
        f"min={city_component.min():.2f}  "
        f"max={city_component.max():.2f}  "
        f"mean={city_component.mean():.2f}"
    )

    #? ── Component 5: is_weekend — 10 points ────────────────────
    # Weekend is a small risk uplift, not a fraud rule by itself.
    weekend_component = df['is_weekend'].fillna(0).astype(int).mul(10)

    print(
        f"  Component 5 (is_weekend)    : "
        f"min={weekend_component.min():.2f}  "
        f"max={weekend_component.max():.2f}  "
        f"mean={weekend_component.mean():.2f}"
    )

    #? ── Final score ────────────────────────────────────────────
    # Add all components together and keep the final score between 0 and 100.
    df['fraud_score'] = (
        amount_component
        + time_component
        + dist_component
        + city_component
        + weekend_component
    ).clip(0, 100).round(2)

    # ── Validation ─────────────────────────────────────────────
    # Fraud transactions should have a higher score than legitimate ones.
    legit_mean = df.loc[df['is_fraud'] == 0, 'fraud_score'].mean()
    fraud_mean = df.loc[df['is_fraud'] == 1, 'fraud_score'].mean()

    legit_median = df.loc[df['is_fraud'] == 0, 'fraud_score'].median()
    fraud_median = df.loc[df['is_fraud'] == 1, 'fraud_score'].median()

    mean_difference = fraud_mean - legit_mean

    print(
        f"\n  Legitimate txns    : mean={legit_mean:.2f}  "
        f"median={legit_median:.2f}"
    )

    print(
        f"  Fraud txns         : mean={fraud_mean:.2f}  "
        f"median={fraud_median:.2f}"
    )

    print(
        f"  Mean difference    : {mean_difference:.2f} "
        f"(fraud mean minus legitimate mean)"
    )

    print(
        f"  Score range        : "
        f"{df['fraud_score'].min():.2f} – {df['fraud_score'].max():.2f}"
    )

    #? ── Score distribution ─────────────────────────────────────
    # This helps check whether fraud transactions are moving into higher buckets.
    print("\n  Score distribution:")

    score_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    score_labels = [
        '0-10', '10-20', '20-30', '30-40', '40-50',
        '50-60', '60-70', '70-80', '80-90', '90-100'
    ]

    score_buckets = pd.cut(
        df['fraud_score'],
        bins=score_bins,
        labels=score_labels,
        include_lowest=True
    )

    for label, count in score_buckets.value_counts().sort_index().items():
        percentage = count / len(df) * 100
        bucket_mask = score_buckets == label
        fraud_count = df.loc[bucket_mask, 'is_fraud'].sum()

        print(
            f"    {label:8s} → {count:>10,}  ({percentage:5.1f}%)  "
            f"fraud: {fraud_count:>5,}"
        )

    #? ── Correlation check ──────────────────────────────────────
    # A positive correlation means higher fraud_score generally aligns with fraud transactions.
    correlation = df['fraud_score'].corr(df['is_fraud'])

    print(f"\n  Correlation with is_fraud  : {correlation:.4f}")

    if correlation > 0.1:
        print("  Validation         : positive correlation confirmed ✓")
    else:
        print("  WARNING: low correlation — review the score weights")

    print("\n  Added              : fraud_score")

    return df


#! ─────────────────────────────────────────────────────────────
#! FEATURE 5 — RISK TIER
#! ─────────────────────────────────────────────────────────────

def add_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a readable risk category from the numeric fraud_score.

    fraud_score is useful because it gives an exact risk value from 0 to 100.
    
    Risk tiers:
        LOW    : 0 to 33
        MEDIUM : greater than 33 to 66
        HIGH   : greater than 66 to 100

    These thresholds split the 0-100 score into three equal groups.
    In a real production system, the thresholds could be tuned based on
    analyst review capacity and acceptable false positive rates.
    """
    print("\nFeature 5: risk_tier")

    # Make sure the columns needed for this feature exist.
    required_cols = ['fraud_score', 'is_fraud']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()

    #? ── Assign risk tier from fraud_score ───────────────────────
    # right=True means the upper boundary is included in each bucket:
    #   33: LOW
    #   66: MEDIUM
    #   67: HIGH
    #
    # include_lowest=True makes sure score 0 is included in LOW.
    df['risk_tier'] = pd.cut(
        df['fraud_score'],
        bins=[0, 33, 66, 100],
        labels=['LOW', 'MEDIUM', 'HIGH'],
        right=True,
        include_lowest=True
    )

    # pd.cut returns a categorical column.
    # I convert it to text because it is easier to export and use in Power BI later.
    
    # If any score falls outside the expected 0-100 range, it becomes UNKNOWN
    # so the issue is visible instead of silently becoming blank.
    df['risk_tier'] = (
        df['risk_tier']
          .astype(str)
          .replace('nan', 'UNKNOWN')
          .str.strip()
    )

    #? ── Check for unexpected scores ─────────────────────────────
    unknown_count = (df['risk_tier'] == 'UNKNOWN').sum()

    if unknown_count > 0:
        print(
            f"  WARNING: {unknown_count:,} scores did not fit into "
            f"LOW / MEDIUM / HIGH. Check fraud_score values."
        )

        print("  Affected fraud_score summary:")
        print(
            df.loc[df['risk_tier'] == 'UNKNOWN', 'fraud_score']
              .describe()
              .round(2)
        )
    else:
        print("  NaN check          : no out-of-range scores ✓")

    #? ── Overall tier distribution ───────────────────────────────
    # This shows how transactions are spread across the three risk levels.
    print("\n  Distribution across all transactions:")

    for tier in ['LOW', 'MEDIUM', 'HIGH']:
        count = (df['risk_tier'] == tier).sum()
        percentage = count / len(df) * 100

        print(f"    {tier:6s} → {count:>10,}  ({percentage:.1f}%)")

    #? ── Fraud capture by tier ───────────────────────────────────
 
    print("\n  Fraud capture rate per tier:")

    total_fraud = df['is_fraud'].sum()

    for tier in ['LOW', 'MEDIUM', 'HIGH']:
        tier_mask = df['risk_tier'] == tier

        tier_total = tier_mask.sum()
        tier_fraud = df.loc[tier_mask, 'is_fraud'].sum()

        fraud_capture_pct = (
            tier_fraud / total_fraud * 100
            if total_fraud > 0 else 0
        )

        fraud_rate = (
            tier_fraud / tier_total * 100
            if tier_total > 0 else 0
        )

        print(
            f"    {tier:6s} → {tier_fraud:>5,} fraud  "
            f"({fraud_capture_pct:.1f}% of all fraud)  "
            f"fraud rate: {fraud_rate:.3f}%"
        )

    #? ── HIGH tier precision ─────────────────────────────────────
    # This answers the question of  how many of them are actually fraud? if comparing only HIGH tier transactions
    high_mask = df['risk_tier'] == 'HIGH'

    high_total = high_mask.sum()
    high_fraud = df.loc[high_mask, 'is_fraud'].sum()

    high_precision = (
        high_fraud / high_total * 100
        if high_total > 0 else 0
    )

    review_ratio = (
        int(high_total / high_fraud)
        if high_fraud > 0 else 'N/A'
    )

    print("\n  HIGH tier precision:")
    print(f"    Total HIGH transactions : {high_total:,}")
    print(f"    Fraud in HIGH tier      : {high_fraud:,}")
    print(f"    Precision               : {high_precision:.2f}%")
    print(f"    Review ratio            : 1 in every {review_ratio} HIGH transactions is fraud")

    #? ── Fraud score stats by tier ───────────────────────────────
    # This validates that each label contains the expected score range.
    print("\n  Fraud score stats per tier:")

    print(
        df.groupby('risk_tier')['fraud_score']
          .agg(['min', 'mean', 'max', 'count'])
          .round(2)
          .to_string()
    )

    print("\n  Added              : risk_tier")

    return df


#! ─────────────────────────────────────────────────────────────
#! SAVE
#! ─────────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame, path: str) -> None:
    """
    Save the feature-engineered dataset to CSV.

    This file is separate from fraud_canadianized.csv.
    fraud_canadianized.csv is the cleaned Canadian version of the data,
    while fraud_features.csv is the analytics-ready version.
    """
    # Create the output folder if it does not already exist.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save without the pandas index because transaction_id already identifies rows.
    df.to_csv(path, index=False)

    print(f"\n  Saved {len(df):,} rows → {path}")
#* ─────────────────────────────────────────────────────────────
#* MAIN
#* ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Load processed CSV — no Faker wait, runs in seconds
    df = load(PROCESSED_PATH)
   # ── Run features built so far ─
    df = add_age_group(df)
    df = add_amount_zscore(df)
    df = add_fraud_type(df)
    df = add_fraud_score(df)
    df = add_risk_tier(df)
 
 # ── Validation checks ─────────────────────────────────────────
    print("\n=== age_group vs fraud ===")
    print(df.groupby(['age_group', 'is_fraud'])
            .size().unstack(fill_value=0))
    print("\n=== amount_zscore: fraud vs legit ===")
    print(df.groupby('is_fraud')['amount_zscore']
            .describe().round(3))
 
    print("\n=== Current columns ===")
    print(f"  Count : {df.shape[1]}")
    for col in df.columns:
        print(f"    {col:25s}  {str(df[col].dtype):15s}  "
              f"sample: {df[col].iloc[0]}")
    print("\n=== fraud_type vs category (fraud only) ===")
    print(df[df['is_fraud'] == 1]
            .groupby(['fraud_type', 'category'])
            .size()
            .sort_values(ascending=False)
            .to_string())

   
    print("\n=== fraud_score: fraud vs legit ===")
    print(df.groupby('is_fraud')['fraud_score']
            .describe().round(2))

    print("\n=== fraud_score by fraud_type ===")
    print(df[df['is_fraud'] == 1]
            .groupby('fraud_type')['fraud_score']
            .describe().round(2))

    
    print("\n=== risk_tier vs fraud ===")
    print(df.groupby(['risk_tier', 'is_fraud'])
            .size().unstack(fill_value=0))

    print("\n=== risk_tier by age_group (fraud only) ===")
    print(df[df['is_fraud'] == 1]
            .groupby(['age_group', 'risk_tier'])
            .size().unstack(fill_value=0))

    print("\n=== Final columns ===")
    print(f"  Count : {df.shape[1]}")
    for col in df.columns:
        print(f"    {col:25s}  {str(df[col].dtype):15s}  "
              f"sample: {df[col].iloc[0]}")
   
    save_features(df, FEATURES_PATH)


 
