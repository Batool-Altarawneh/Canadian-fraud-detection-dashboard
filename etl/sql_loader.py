# Loads the cleaned dimension and fact CSV files into SQL Server FraudDB.

# I load the dimension tables first because FactTransaction has foreign keys that depend on them. 
# If the fact table is loaded before the dimensions,SQL Server will reject the rows because the referenced keys do not exist yet.

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()



DIMS_PATH = '../data/processed/'


TABLES_IN_ORDER = [
    'DimCustomer',
    'DimMerchant',
    'DimDate',
    'DimAlertType',
    'FactTransaction',
]

#! ─────────────────────────────────────────────────────────────
#! Build the Connection Engine
#! ─────────────────────────────────────────────────────────────
def get_engine():
    """
    Create and return a SQLAlchemy engine for SQL Server.

    This project uses Windows Authentication, so I do not need to storea database username or password in the code. 
    Instead, SQL Server will use the current Windows user account to connect.

    The server name and database name are read from the .env file so the connection settings stay outside the code and can be changed easily.
    """

    # Read the SQL Server connection values from the .env file.
   
    server = os.getenv('SQL_SERVER')
    db = os.getenv('SQL_DATABASE')

    # Stop the script early if the required .env values are missing.
    if not server or not db:
        raise ValueError(
            "SQL_SERVER and SQL_DATABASE must be set in your .env file"
        )

    # Build the SQLAlchemy connection string for SQL Server through pyodbc.
    #
    # mssql+pyodbc tells SQLAlchemy to connect to Microsoft SQL Server using the pyodbc driver.
    # trusted_connection=yes means Windows Authentication is used instead of a username and password.

    conn_str = (
        f"mssql+pyodbc://{server}/{db}"
        f"?driver=ODBC+Driver+17+for+SQL+Server"
        f"&trusted_connection=yes"
    )

    # Create the SQLAlchemy engine.
    #
    # fast_executemany=True helps speed up bulk inserts when loading many rows from the CSV files into SQL Server.
    engine = create_engine(conn_str, fast_executemany=True)


    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    print(f"  Connected to {server} / {db}")

    return engine


#! ─────────────────────────────────────────────────────────────
#! Load Each CSV with Type Coercion
#! ─────────────────────────────────────────────────────────────

def load_csv(table_name: str) -> pd.DataFrame:
    """
    Load one CSV file and prepare its columns for SQL Server.

    Each CSV file should have the same name as its target SQL table.
    Example:
        DimCustomer.csv     : DimCustomer table
        DimMerchant.csv     : DimMerchant table
        DimDate.csv         : DimDate table
        DimAlertType.csv    : DimAlertType table
        FactTransaction.csv : FactTransaction table

    This function also fixes a few data types before loading:
        - BIT columns are converted to 0/1
        - alert_key keeps NULL values for non-fraud transactions
        - full_date is converted to a real date value
    """

   
    path = os.path.join(DIMS_PATH, f'{table_name}.csv')

    df = pd.read_csv(path)

    # Print a small summary so I can confirm the correct file is being loaded.
    print(f"\n  Loading {table_name} from CSV...")
    print(f"    Rows    : {len(df):,}")
    print(f"    Columns : {list(df.columns)}")

    # SQL Server BIT columns expect 0/1 values.
    # But because in pandas, these columns may appear as True/False, so I convert them to integers before inserting into SQL Server.
    bit_cols = [
        'is_fraud',
        'is_weekend',
        'is_high_risk_category',
        'is_holiday_ca'
    ]

    for col in bit_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # alert_key is nullable in FactTransaction.
    #Legitimate transactions do not have a fraud alert, so their alert_key should stay NULL in SQL Server.

    if 'alert_key' in df.columns:
        df['alert_key'] = df['alert_key'].astype('Int64')

    # full_date belongs to DimDate.
    #
    # I convert it from text into a Python date object so SQL Server can load
    # it cleanly into the DATE column.
    if 'full_date' in df.columns:
        df['full_date'] = pd.to_datetime(df['full_date']).dt.date

    return df

#! ─────────────────────────────────────────────────────────────
#! Insert into SQL Server
#! ─────────────────────────────────────────────────────────────

def insert_table(df: pd.DataFrame, table_name: str, engine) -> None:
    """
    Insert one DataFrame into its matching SQL Server table.

    The table must already exist in SQL Server because this script is responsible for loading data.

   
    """

    # Load the DataFrame into SQL Server.
    #
    # if_exists='append':
    #   - do not drop the existing table
    #   - do not recreate the table
    #   - only add rows into the table
    #
    # This is important because my tables were already created using the DDL script.
    # And I want SQL Server to keep the primary keys, foreign keys, data types, defaults, and constraints from that script.
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='append',
        index=False, # not allowing to insert the DataFrame index as an extra SQL col
        chunksize=1000, # it avoids sending the entire DataFrame to SQL Server in one huge insert
        method='multi',
    )

    print(f" Inserted {len(df):,} rows → {table_name}")
#! ─────────────────────────────────────────────────────────────
#!  Row Count Validation
#! ─────────────────────────────────────────────────────────────

def validate_row_counts(engine) -> None:
    """
    Compare the number of rows in each CSV file with the number of rows loaded into SQL Server.

    This is useful validation step after the load finishes.
    If the CSV count and SQL count match, it means the table received the
    expected number of records.

    If they do not match, I know I need to check the insert step, table constraints, or possible duplicate data.
    """

    print("\n=== Row Count Validation ===")

    # Open one database connection and reuse it while checking all tables.
    with engine.connect() as conn:

        # Check the tables in the same order used during the load.
        for table in TABLES_IN_ORDER:

            # Count how many rows currently exist in the SQL Server table.
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            sql_count = result.scalar()

            # Count how many rows exist in the original CSV file.
            # The CSV file name should match the table name.
            csv_path = os.path.join(DIMS_PATH, f"{table}.csv")
            csv_count = len(pd.read_csv(csv_path))

            # Show a clear success/mismatch status for each table.
            status = "Done" if sql_count == csv_count else " MISMATCH"

            print(
                f"  {table:<20} "
                f"CSV: {csv_count:>8,}   "
                f"SQL: {sql_count:>8,}   "
                f"{status}"
            )

#* ─────────────────────────────────────────────────────────────
#* MAIN
#* ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    
    print("Connecting to SQL Server...")
    engine = get_engine()
    for table_name in TABLES_IN_ORDER:
        df = load_csv(table_name)
        insert_table(df, table_name, engine)

    validate_row_counts(engine)

    print("\n All tables loaded successfully into FraudDB")
