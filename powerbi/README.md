# Power BI report

`RetentionDashboard.pbip` is a **Power BI project** (the text-based format of a `.pbix`): the semantic model is in
`RetentionDashboard.SemanticModel/` (TMDL files) and the report pages in `RetentionDashboard.Report/`. Screenshots and a PDF
export are in [`../reports/`](../reports).

## Open it

1. Install Power BI Desktop (free, Windows) and open `RetentionDashboard.pbip`.
2. *Transform data → Edit parameters*: set **DataFolder** to the `reports\` folder of your clone (ending with `\`).
3. *Home → Refresh*. The report reads the small CSV files in `reports/dashboard/` and `reports/coupon_list.csv`, so the raw data is not needed.

## Pages

| Page | Question it answers | Visuals |
|---|---|---|
| Business overview | How is the store doing week by week? | 4 KPI cards, week slicer, net sales and active households per week, retention per 4-week period, top departments |
| Customer segments | Where do sales come from? | share of households vs share of sales per RFM segment, segment table |
| Churn & coupons | Who should get a retention coupon? | churn rate per snapshot, coupon list sorted by expected sales lost |

## Data model

| Table | Grain | Source |
|---|---|---|
| `weekly_kpis` | one row per week | `sql/07_weekly_kpis.sql` |
| `period_retention` | one row per 4-week period | `sql/03_period_retention.sql` |
| `dim_week` | one row per week, with its 4-week `period` | DAX calculated table |
| `department_sales`, `segments`, `churn_by_snapshot`, `coupon_list` | one row per department / segment / cutoff week / household | `src/dashboard_data.py`, `src/predict.py` |

Relationships: `weekly_kpis[week_no] → dim_week[week_no]`, and `dim_week[period] → period_retention[period]` in **both
directions**, so the week slicer filters the weekly charts *and* the retention chart. Weeks 1-17 are removed with a page filter
on `dim_week[phase]` (households were still joining the panel).

## Measures (DAX)

| Measure | DAX idea |
|---|---|
| Net Sales | `SUM(net_sales)` |
| Active Households | `MEDIAN(active_households)`: a typical week |
| Basket Value | `DIVIDE(SUM(net_sales), SUM(baskets))` |
| Retention Rate | `DIVIDE(SUM(retained_next_period), SUM(active_households))`: pooled over the selected periods |
| Segment Share of Sales | `DIVIDE(SUM(sales), CALCULATE(SUM(sales), ALL(segments)))`: share of the grand total |
| Latest Churn Rate, Churn Change (pts) | `VAR` for the last and previous cutoff week, then `CALCULATE` at each one |
| Expected Sales Lost | `SUM(expected_sales_lost)` on the coupon list |

`department_sales[sales_rank]` is a calculated column (`RANKX`) used to keep the top 6 departments. The values were checked
against the Python results: net sales 7,377,567 (weeks 18-102), 1,318 active households, basket 29.33, retention 89.8%,
churn 8.2% (+1.4 pts), 231 households on the list.
