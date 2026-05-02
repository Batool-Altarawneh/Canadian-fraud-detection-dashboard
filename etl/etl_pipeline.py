import pandas as pd
import numpy as np
from faker import Faker
from scipy import stats
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()
fake = Faker('en_CA')
Faker.seed(42)   # makes Faker results reproducible
np.random.seed(42)


RAW_PATH       = '../data/raw/fraudTrain.csv'
PROCESSED_PATH = '../data/processed/fraud_canadianized.csv'
FX_RATE        = 1.37   # USD to CAD


# ############################################################
# Lookup Tables
# ############################################################
# US state codes → Canadian province codes
# US state codes mapped to similar Canadian province codes
#
# Canadian province abbreviations:
# AB = Alberta
# BC = British Columbia
# MB = Manitoba
# SK = Saskatchewan
# ON = Ontario
# QC = Quebec
# NB = New Brunswick
# NS = Nova Scotia
#
# US state abbreviations used below:
# AL = Alabama, AK = Alaska, AZ = Arizona, AR = Arkansas, CA = California
# CO = Colorado, CT = Connecticut, DE = Delaware, FL = Florida, GA = Georgia
# HI = Hawaii, ID = Idaho, IL = Illinois, IN = Indiana, IA = Iowa
# KS = Kansas, KY = Kentucky, LA = Louisiana, ME = Maine, MD = Maryland
# MA = Massachusetts, MI = Michigan, MN = Minnesota, MS = Mississippi
# MO = Missouri, MT = Montana, NE = Nebraska, NV = Nevada
# NH = New Hampshire, NJ = New Jersey, NM = New Mexico, NY = New York
# NC = North Carolina, ND = North Dakota, OH = Ohio, OK = Oklahoma
# OR = Oregon, PA = Pennsylvania, RI = Rhode Island, SC = South Carolina
# SD = South Dakota, TN = Tennessee, TX = Texas, UT = Utah
# VT = Vermont, VA = Virginia, WA = Washington, WV = West Virginia
# WI = Wisconsin, WY = Wyoming, DC = District of Columbia

PROVINCE_MAP = {
    # Western Canada
    'AK': 'BC',
    'CA': 'BC',
    'HI': 'BC',
    'NV': 'BC',
    'OR': 'BC',
    'WA': 'BC',

    # Alberta
    'AL': 'AB',
    'AZ': 'AB',
    'CO': 'AB',
    'ID': 'AB',
    'MT': 'AB',
    'NM': 'AB',
    'OK': 'AB',
    'TX': 'AB',
    'UT': 'AB',
    'WY': 'AB',

    # Prairies | Manitoba
    'AR': 'MB',
    'IA': 'MB',
    'MN': 'MB',
    'WI': 'MB',

    # Prairies | Saskatchewan
    'KS': 'SK',
    'NE': 'SK',
    'ND': 'SK',
    'SD': 'SK',

    # Ontario
    'CT': 'ON',
    'FL': 'ON',
    'GA': 'ON',
    'IL': 'ON',
    'IN': 'ON',
    'KY': 'ON',
    'MD': 'ON',
    'MA': 'ON',
    'MI': 'ON',
    'MO': 'ON',
    'NJ': 'ON',
    'NY': 'ON',
    'NC': 'ON',
    'OH': 'ON',
    'PA': 'ON',
    'TN': 'ON',
    'VA': 'ON',
    'WV': 'ON',
    'DC': 'ON',

    # Quebec
    'LA': 'QC',
    'MS': 'QC',

    # Atlantic Canada
    'DE': 'NB',
    'ME': 'NB',
    'VT': 'NB',
    # Atlantic Canada | Nova Scotia
    'NH': 'NS',
    'RI': 'NS',
    'SC': 'NS',
}

# ############################################################
# Province Coordinates
# ############################################################
# Approximate latitude and longitude points for Canadian provinces.

# These values were checked manually using:
# https://www.findlatitudeandlongitude.com/

# The site returns an approximate latitude/longitude pair when a provincen name is searched.
# These points are useful for placing province-level transactions on a map

# Format:
# province_code: (latitude, longitude)

PROVINCE_COORDS = {
    # Western Canada
    'AB': (55.001251, -115.002136),  # Alberta, Canada
    'BC': (55.001251, -125.002441),  # British Columbia, Canada

    # Prairie provinces
    'MB': (55.001251, -97.001038),   # Manitoba, Canada
    'SK': (55.532126, -106.141224),  # Saskatchewan, Canada

    # Central Canada
    'ON': (50.000678, -86.000977),   # Ontario, Canada
    'QC': (52.476089, -71.825867),   # Quebec, Canada

    # Atlantic Canada
    'NB': (46.500283, -66.750183),   # New Brunswick, Canada
    'NS': (45.196040, -63.165379),   # Nova Scotia, Canada

    'PE': (46.335551, -63.146668),             # Prince Edward Island, Canada
    'NL': (53.821733, -61.229553),             # Newfoundland and Labrador, Canada
}

# Fallback point used when the province code is missing or unknown.
# This keeps the script running instead of failing on an unmapped province.
DEFAULT_COORD = (50.0, -95.0)


# ############################################################
# Canadian Merchant Names
# ############################################################
# This lookup table gives each transaction category a list of Canadian or Canada-operating merchant names.

# Why we need it:
# The original fraud dataset contains US-style merchant names. Since this project Canadianizes the dataset, these names make the transactions look more realistic for a Canadian context.

# Notes:
#pos : point-of-sale 
#net : online transactions.
# Some companies are Canadian-owned, while others are international brands that operate in Canada.


CA_MERCHANTS = {
    # Gas stations and fuel-related merchants.
    'gas_transport': [
        'Petro-Canada',
        'Esso',
        'Shell Canada',
        'Irving Oil',
        'Ultramar',
    ],

    # In-person grocery transactions.
    'grocery_pos': [
        'Loblaws',
        'Metro',
        'Sobeys',
        'No Frills',
        'FreshCo',
        'IGA',
        'Real Canadian Superstore',
    ],

    # Online grocery transactions.
    'grocery_net': [
        'Amazon Fresh Canada',
        'Goodfood',
        'HelloFresh Canada',
        'Instacart Canada',
        'Walmart Grocery Canada',
    ],

    # Home improvement, furniture, and household-related merchants.
    'home': [
        'Home Depot Canada',
        'IKEA Canada',
        'Canadian Tire',
        'Rona',
        'Wayfair Canada',
    ],

    # In-person shopping transactions.
    'shopping_pos': [
        "Hudson's Bay",
        'Sport Chek',
        'Winners',
        'HomeSense',
        'Reitmans',
    ],

    # Online shopping transactions.
    'shopping_net': [
        'Amazon.ca',
        'Best Buy Canada',
        'Indigo',
        'Sport Chek Online',
        'Apple Canada',
    ],

    # Kids, toys, education, and pet-related merchants.
    'kids_pets': [
        'PetSmart Canada',
        'Mastermind Toys',
        "Scholar's Choice",
        'Global Pet Foods',
    ],

    # Entertainment and digital media merchants.
    'entertainment': [
        'Cineplex',
        'Apple Canada',
        'Spotify Canada',
        'Rogers Media',
        'Bell Media',
    ],

    # Restaurants, fast food, and food delivery merchants.
    'food_dining': [
        'Tim Hortons',
        'A&W Canada',
        "Harvey's",
        'Swiss Chalet',
        'Boston Pizza',
        'St-Hubert',
        "McDonald's Canada",
        'Subway Canada',
        "Domino's Canada",
        'DoorDash Canada',
    ],

    # Pharmacies, fitness, and health-related merchants.
    'health_fitness': [
        'Shoppers Drug Mart',
        'Rexall',
        'Jean Coutu',
        'GoodLife Fitness',
        'Pharmasave',
    ],

    # Beauty, personal care, and self-care merchants.
    'personal_care': [
        'Shoppers Drug Mart',
        'Sephora Canada',
        'Lush Canada',
        'Bath & Body Works Canada',
    ],

    # Travel, transportation, accommodation, and car rental merchants.
    'travel': [
        'Air Canada',
        'WestJet',
        'VIA Rail',
        'Enterprise Canada',
        'Airbnb Canada',
    ],

    # Miscellaneous in-person retail.
    'misc_pos': [
        'Dollarama',
        'Giant Tiger',
        'Bulk Barn',
        'The Source',
        'Staples Canada',
    ],

    # Miscellaneous online retail.
    'misc_net': [
        'Amazon.ca',
        'eBay Canada',
        'Etsy Canada',
        'Rakuten Canada',
        'Wish',
    ],
}

#? ─────────────────────────────────────────────────────────────
#? Helper Functions
#? ─────────────────────────────────────────────────────────────

def assign_city_tier(pop: float) -> str:
    """
    Turn a city's population into a simple size category.
    Instead of keeping population only as a raw number, this function creates an easier label to understand and analyze.
    For example, it is much easier to compare fraud patterns by (Metro) or (Small Town) than by thousands of different population values.
    
    - Metro       : 1,000,000 or more people
    - Large City  : 100,000 to 999,999 people
    - Mid City    : 30,000 to 99,999 people
    - Small Town  : less than 30,000 people
    """

    if pop >= 1_000_000:
        return 'Metro'
    elif pop >= 100_000:
        return 'Large City'
    elif pop >= 30_000:
        return 'Mid City'
    else:
        return 'Small Town'


def hour_to_period(h: int) -> str:
    """
   Turn the transaction hour into a clear time-of-day label.
   Instead of looking at raw numbers like 2, 9, or 17, this helps us read the data in a more natural way, such as Late Night, Morning Afternoon, or Evening.
   
   This is useful for fraud analysis because some unusual transactions may happen more often during certain parts of the day, especially outside normal activity hours.
   
   Expected input:
    h should be an hour number from 0 to 23.
    """
    if 0 <= h < 6:
        return 'Late Night'
    elif 6 <= h < 12:
        return 'Morning'
    elif 12 <= h < 18:
        return 'Afternoon'
    elif 18 <= h <= 23:
        return 'Evening'
    else:
        return 'Unknown'
    

#! ─────────────────────────────────────────────────────────────
#! STEP 1 — INGEST
#! ─────────────────────────────────────────────────────────────

def ingest(path: str) -> pd.DataFrame:
    """
    Load the raw fraud CSV file and print a quick data quality summary.

    This is the first step in the pipeline. Before cleaning or changing anything, we load the source data and check the basic details:
        - how many rows and columns we have?
        - how many transactions are marked as fraud?
        - whether there are missing values?
        - the date range covered by the dataset?

    These checks help confirm that the file loaded correctly and  the data looks reasonable before moving to the next ETL steps.

    """

    print("Loading data...")

    df = pd.read_csv(
        path,
        parse_dates=['trans_date_trans_time']
    )

    print(f"  Rows      : {len(df):,}")
    print(f"  Columns   : {df.shape[1]}")
    print(f"  Fraud     : {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.4%})")
    print(f"  Nulls     : {df.isnull().sum().sum()}")
    print(
        f"  Date range: {df['trans_date_trans_time'].min().date()} "
        f"to {df['trans_date_trans_time'].max().date()}"
    )

    return df


#! ─────────────────────────────────────────────────────────────
#! STEP 2 — CLEAN
#! ─────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw transaction data and prepare it for the Canadian version.

    In this step:
        - remove duplicate transactions
        - fix important column data types
        - clean basic text fields
        - create province_code from the original US state column
        - create city_tier from the original city population column
        - remove US-specific location columns after we no longer need them

    """

    print("\nCleaning...")

    # Remove duplicate transactions based on the transaction number.
    # trans_num is treated as the unique transaction identifier.
    before = len(df)
    df = df.drop_duplicates(subset='trans_num')
    print(f"  Duplicates removed : {before - len(df):,}")

    # cc_num is kept as text because credit card numbers should not be used
    # for math, and keeping them as strings avoids formatting issues.
    df['cc_num'] = df['cc_num'].astype(str)
    df['is_fraud'] = df['is_fraud'].astype(int)

    # Standardize text columns so grouping and filtering work consistently.
    df['merchant'] = df['merchant'].str.strip()
    df['category'] = df['category'].str.strip().str.lower()
    df['gender'] = df['gender'].str.upper().str.strip()

    # Create the Canadian province code while the original US state column
    # is still available. If a state does not map for any reason, default to ON.
    df['province_code'] = df['state'].map(PROVINCE_MAP).fillna('ON')

    # Convert raw city population into an easier-to-analyze city size group.
    # This is more useful in dashboards than showing population as a raw number.
    df['city_tier'] = df['city_pop'].apply(assign_city_tier)

    print(f"  City tiers         : {df['city_tier'].value_counts().to_dict()}")

    # Remove original US location fields after saving the Canadian replacements.
    
    df = df.drop(
        columns=[
            'Unnamed: 0',
            'zip',
            'street',
            'city',
            'state',
            'lat',
            'long',
            'merch_lat',
            'merch_long',
            'city_pop',
        ],
        errors='ignore'
    )

    print(f"  Columns after drop : {df.shape[1]}")
    print(f"  Rows after clean   : {len(df):,}")

    return df

#! ─────────────────────────────────────────────────────────────
#! STEP 3 — CANADIANIZE
#! ─────────────────────────────────────────────────────────────

def canadianize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the cleaned dataset into a Canadian-style version.

    In this step, I replace or create fields that make the dataset more realistic for a Canadian fraud-detection project:
        - generate Canadian city names and postal codes
        - create customer coordinates based on the mapped province
        - create merchant coordinates with a separate random offset
        - calculate the approximate distance between customer and merchant
        - convert transaction amounts from USD to CAD
        - replace merchant names with Canadian or Canada-operating merchants

    """

    print("\nCanadianizing...")

    # ── Canadian cities and postal codes ──────────────────────────
    # Faker is used here to generate realistic Canadian-style city names and postal codes.
    # These are synthetic values, not real customer addresses.
    print("  Generating Canadian locations...")

    df['city_ca'] = [fake.city() for _ in range(len(df))]
    df['postal_code'] = [fake.postalcode() for _ in range(len(df))]

    # Standardize postal codes by removing spaces and forcing uppercase.
    # Example: "K1A 0B1" becomes "K1A0B1"
    df['postal_code'] = df['postal_code'].str.replace(' ', '').str.upper()

    # ── Customer coordinates ──────────────────────────────────────
    # Start from the province's approximate coordinate, then add a small random offset so not every customer appears at the exact same point.
    base_lat = df['province_code'].map(
        lambda p: PROVINCE_COORDS.get(p, DEFAULT_COORD)[0]
    )

    base_long = df['province_code'].map(
        lambda p: PROVINCE_COORDS.get(p, DEFAULT_COORD)[1]
    )

    # ±2 degrees keeps the customer location around the selected province area.
    df['lat'] = (
        base_lat + np.random.uniform(-2.0, 2.0, size=len(df))
    ).round(6)

    df['long'] = (
        base_long + np.random.uniform(-2.0, 2.0, size=len(df))
    ).round(6)

    # ── Merchant coordinates ──────────────────────────────────────
    # Merchants get their own random offset. This makes the customer and merchant locations different, which allows us to calculate distance.
    # A slightly wider range is used for merchants to simulate purchases
    # from different cities or nearby regions.
    df['merch_lat'] = (
        base_lat + np.random.uniform(-3.0, 3.0, size=len(df))
    ).round(6)

    df['merch_long'] = (
        base_long + np.random.uniform(-3.0, 3.0, size=len(df))
    ).round(6)

    # ── Distance between customer and merchant ────────────────────
    # This gives an approximate distance in kilometres.
    
    # Example:
    # - 1 degree of latitude is about 111 km
    # - longitude distance changes depending on latitude, so we adjust it using cosine
    # https://www.sco.wisc.edu/2022/01/21/how-big-is-a-degree/
    
    # - then we use the Pythagorean formula to estimate the final distance
    # https://www.geeksforgeeks.org/dsa/program-calculate-distance-two-points/
    # https://stackoverflow.com/questions/57294120/calculating-distance-between-latitude-and-longitude-in-python
    # This is accurate enough for synthetic fraud-analysis features.
    df['dist_km'] = np.sqrt(
        ((df['lat'] - df['merch_lat']) * 111.0) ** 2 +
        (
            (df['long'] - df['merch_long'])
            * 111.0
            * np.cos(np.radians(df['lat']))
        ) ** 2
    ).round(2)

    print("  Coordinates        : customer + merchant, province-based")
    print(
        f"  Distance avg       : {df['dist_km'].mean():.1f} km "
        f"| max: {df['dist_km'].max():.1f} km"
    )

    # ── Amount conversion: USD to CAD ─────────────────────────────
    # The original dataset amount is in USD. For the Canadian version,I convert it to CAD using the fixed FX_RATE defined at the top.
    df['amount_cad'] = (df['amt'] * FX_RATE).round(2)

    print(f"  Amount CAD avg     : ${df['amount_cad'].mean():.2f}")

    # ── Canadian merchant names ──────────────────────────────────
    # Replace the original merchant name with a Canadian-style merchant based on the transaction category.
    
    # If a category is missing from CA_MERCHANTS, use a safe default name
   
    df['merchant_ca'] = df['category'].apply(
        lambda c: np.random.choice(
            CA_MERCHANTS.get(c, ['Generic Canadian Merchant'])
        )
    )

    print(
        f"  Provinces (top 3)  : "
        f"{df['province_code'].value_counts().head(3).to_dict()}"
    )

    return df


#! ─────────────────────────────────────────────────────────────
#! STEP 4 — BASE FEATURE ENGINEERING
#! ─────────────────────────────────────────────────────────────

def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the first set of useful features for analysis and modeling.

    These are called "base features" because they are simple, and useful across the whole project.

    Features added here:
        1. Security features
           - mask the credit card number using SHA-256

        2. Time features
           - transaction hour
           - day name
           - month name
           - weekend flag
           - time-of-day label

        3. Amount features
           - flag unusually high or low transaction amounts using IQR
           - group transaction amounts into readable bins
    """

    print("\nEngineering base features...")

    # ── 1. Mask credit card number ────────────────────────────────
    # Credit card numbers are sensitive, so we should not keep them in plain text. 
    # SHA-256 turns each card number into a fixed hash.
    #
    # This keeps the value useful for grouping the same customer/card, but protects the original card number.
    df['cc_num_masked'] = df['cc_num'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()
    )

    print("  CC numbers masked  : SHA-256")

    # ── 2. Time features ─────────────────────────────────────────
    # Break the transaction timestamp into easier-to-analyze fields.
    # These are useful for dashboards and fraud pattern detection.
    df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
    df['transaction_day'] = df['trans_date_trans_time'].dt.day_name()
    df['transaction_month'] = df['trans_date_trans_time'].dt.month_name()

    # Weekend flag:
    # weekday returns Monday=0 through Sunday=6.
    # So Saturday=5 and Sunday=6.
    df['is_weekend'] = (
        df['trans_date_trans_time']
        .dt.weekday
        .isin([5, 6])
        .astype(int)
    )

    # Convert raw hour into labels like Morning, Afternoon, Evening, etc.
    df['time_of_day'] = df['transaction_hour'].apply(hour_to_period)

    print("  Time features      : hour, day, month, is_weekend, time_of_day")

    # ── 3a. Amount outlier flag using IQR ─────────────────────────
    # IQR helps identify transactions that are unusually small or large
    # compared with the normal transaction amount range.
    #
    # Formula:
    # IQR = Q3 - Q1
    # Lower bound = Q1 - 1.5 * IQR
    # Upper bound = Q3 + 1.5 * IQR
    #
    # Any amount outside this range is marked as an outlier.
    q1 = df['amount_cad'].quantile(0.25)
    q3 = df['amount_cad'].quantile(0.75)
    iqr = q3 - q1

    # Amounts cannot be negative, so the lower bound is capped at 0.
    lower = max(0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr

    df['is_amount_outlier'] = (
        (df['amount_cad'] < lower) |
        (df['amount_cad'] > upper)
    ).astype(int)

    outlier_count = df['is_amount_outlier'].sum()
    fraud_outlier = df[df['is_fraud'] == 1]['is_amount_outlier'].sum()

    # Calculate the percentage of fraud transactions that were also amount outliers.
    # The if-statement prevents division by zero if the dataset has no fraud rows.
    fraud_total = df['is_fraud'].sum()

    if fraud_total > 0:
        fraud_outlier_pct = fraud_outlier / fraud_total
    else:
        fraud_outlier_pct = 0

    print(f"  IQR bounds         : ${lower:.2f} – ${upper:.2f}")
    print(f"  Outliers flagged   : {outlier_count:,} total")
    print(
        f"  Outliers in fraud  : {fraud_outlier:,} "
        f"({fraud_outlier_pct:.1%} of all fraud)"
    )

    # ── 3b. Amount bins ──────────────────────────────────────────
    # Group transaction amounts into readable ranges.
    # These bins are helpful for Power BI slicers, charts, and summaries.
    df['amount_bin'] = pd.cut(
        df['amount_cad'],
        bins=[0, 20, 50, 100, 200, 500, float('inf')],
        labels=[
            '$0-20',
            '$20-50',
            '$50-100',
            '$100-200',
            '$200-500',
            '$500+'
        ],
        right=True
    ).astype(str)

    print(
        "  Amount bins        : "
        "$0-20 | $20-50 | $50-100 | $100-200 | $200-500 | $500+"
    )

    return df

#! ─────────────────────────────────────────────────────────────
#! STEP 5 — SAVE
#! ─────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, path: str) -> None:
    """
    Save the final processed dataset as a CSV file.

    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)

    print(f"\n  Saved {len(df):,} rows → {path}")

#* ─────────────────────────────────────────────────────────────
#* MAIN
#* ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Run each pipeline step in order.
    df = ingest(RAW_PATH)
    df = clean(df)
    df = canadianize(df)
    df = engineer_base_features(df)

    # ── Final cleanup before saving ───────────────────────────────
    # Drop columns that were replaced, contain sensitive information, or are not needed in the final star-schema dataset.
    
    # These fields are removed because cleaner replacements already exist:
    #cc_num : replaced by cc_num_masked
    #merchant : replaced by merchant_ca
    #amt      : replaced by amount_cad
    
    # Some personal fields are also removed because they are not needed
    # for the final analysis dataset.
    df = df.drop(
        columns=[
            'cc_num',
            'merchant',
            'amt',
            'first',
            'last',
            'job',
            'unix_time',
        ],
        errors='ignore'
    )

    # ── Create transaction_id ─────────────────────────────────────
    # After cleaning and removing duplicates, reset the index and save it as a clean integer transaction_id.
    #  This gives each row a simple surrogate key that can be used later in the database fact table.
    df = df.reset_index(drop=True)
    df.index.name = 'transaction_id'
    df = df.reset_index()

    # ── Save the processed file ───────────────────────────────────
    save_processed(df, PROCESSED_PATH)

    # ── Final validation checks ───────────────────────────────────
    # These prints are not part of the dataset itself. They are quick checks to confirm the pipeline produced the expected output.
    print("\n=== Final columns ===")
    print(f"  Count : {df.shape[1]}")

    for col in df.columns:
        print(f"    {col}")

    print("\n=== Sample row (first transaction) ===")
    print(df.iloc[0].to_string())

    print("\n=== Time features sample ===")
    print(
        df[
            [
                'trans_date_trans_time',
                'transaction_hour',
                'time_of_day',
                'transaction_day',
                'transaction_month',
                'is_weekend',
            ]
        ]
        .head(5)
        .to_string()
    )

    print("\n=== Province distribution ===")
    print(df['province_code'].value_counts().to_string())

    print("\n=== City tier distribution ===")
    print(df['city_tier'].value_counts().to_string())

    print("\n=== Amount bin distribution ===")
    print(df['amount_bin'].value_counts().sort_index().to_string())

    print("\n=== Distance stats (customer vs merchant) ===")
    print(df['dist_km'].describe().round(2).to_string())

    print("\n=== Outlier amounts vs fraud ===")
    print(
        df.groupby(['is_amount_outlier', 'is_fraud'])
        .size()
        .unstack(fill_value=0)
    )

    print("\n=== Merchant sample by category ===")
    print(df.groupby('category')['merchant_ca'].first().to_string())
