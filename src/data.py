"""Load the raw CSV files, check data quality, and run the SQL files with DuckDB.

Usage (from the repo root):
    python -m src.data
This runs every file in sql/ and saves each result to data/processed/<name>.csv
"""

import duckdb
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, SQL_DIR

TABLES = ["transaction_data", "product", "hh_demographic"]


def connect(raw_dir=RAW_DIR) -> duckdb.DuckDBPyConnection:
    """DuckDB connection with one view per raw CSV, so SQL can use `FROM transaction_data`."""
    con = duckdb.connect()
    for name in TABLES:
        path = (raw_dir / f"{name}.csv").as_posix()
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{path}')")
    return con


def run_sql(sql_file, con) -> pd.DataFrame:
    """Run one .sql file and return the result with lower-case column names."""
    df = con.sql(sql_file.read_text(encoding="utf-8")).df()
    df.columns = df.columns.str.lower()
    return df


def load_transactions(raw_dir=RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "transaction_data.csv")
    df.columns = df.columns.str.lower()
    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing values, number of unique values, min and max."""
    numeric = df.select_dtypes("number")
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "n_unique": df.nunique(),
        "min": numeric.min(),
        "max": numeric.max(),
    })


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()
    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        name = sql_file.stem.split("_", 1)[1]          # "01_weekly_activity" -> "weekly_activity"
        df = run_sql(sql_file, con)
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"{sql_file.name:32s} -> {name}.csv ({len(df):,} rows)")


if __name__ == "__main__":
    main()
