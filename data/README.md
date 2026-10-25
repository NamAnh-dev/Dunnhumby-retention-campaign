# Data

Source: **dunnhumby - The Complete Journey** (retail transactions of ~2,500 households over about two years), as distributed on Kaggle.
Check the dataset page for the current license/terms before reuse. Raw data is **not** committed to this repo.

## How to get it

1. Download the dataset from Kaggle and unzip it.
2. Put the CSV files in `data/raw/`.
3. This project only uses:

| File | Used for |
|---|---|
| `transaction_data.csv` | core table: one row per product line in a basket |
| `product.csv` | department lookup (sales by department, departments per household) |
| `hh_demographic.csv` | age, income, household size, home ownership (only ~1/3 of households) |

The other files (campaigns, coupons, `causal_data.csv`) are not used in this version.

## Use the files

From the repo root:

```bash
python -m src.data
```

This runs every SQL file in `sql/` over the three CSV files (with DuckDB) and saves the results in `data/processed/`.
The data-quality checks are in `notebooks/01_eda_sql.ipynb`.

## Notes on the data (verify each one in the EDA notebook)

- `transaction_data` grain is one row per product line, not per basket. The same `basket_id` can contain many rows.
- Rows with zero or negative `quantity` / `sales_value` may exist (voids, returns, coupon-only lines). Decide explicitly how to treat them; do not drop silently.
- Demographics cover only part of the households, so using them as features means handling heavy missingness.
- `day` and `week_no` are relative indexes, not calendar dates. `week_no` is **not** exactly `ceil(day / 7)` (202 of 711 days differ), so weeks are always taken from the data, never computed from `day`.
