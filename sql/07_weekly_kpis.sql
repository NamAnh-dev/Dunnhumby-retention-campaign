-- Store-level KPIs per week, for the dashboard: how many households shop, how much they spend, how much is discounted.
-- Step 1 (CTE): one row per household and week. Step 2: sum the households of each week.
-- Same rules as everywhere else: a household is active in a week only if its net sales are > 0,
-- and discounts are stored as negative numbers in the raw data, so we flip the sign.

WITH household_week AS (
    SELECT
        household_key,
        week_no,
        COUNT(DISTINCT basket_id)                               AS n_baskets,
        SUM(sales_value)                                        AS net_sales,
        -SUM(retail_disc + coupon_disc + coupon_match_disc)     AS discount_amount
    FROM transaction_data
    GROUP BY household_key, week_no
)
SELECT
    week_no,
    COUNT(CASE WHEN net_sales > 0 THEN 1 END)                            AS active_households,   -- COUNT skips NULLs
    CAST(SUM(n_baskets) AS INTEGER)                                      AS baskets,
    SUM(net_sales)                                                       AS net_sales,
    SUM(net_sales) / SUM(n_baskets)                                      AS avg_basket_value,
    SUM(discount_amount) / (SUM(net_sales) + SUM(discount_amount))       AS discount_share
FROM household_week
GROUP BY week_no
ORDER BY week_no
