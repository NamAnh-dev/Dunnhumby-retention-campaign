-- Household x week activity: one row per household and week with at least one transaction.
-- transaction_data has one row per product line, so we aggregate to avoid counting a basket many times.
-- Returns are kept (net_sales can be <= 0); a week only counts as "active" later if net_sales > 0.
-- Discounts are stored as negative numbers in the raw data, so we flip the sign.

SELECT
    t.household_key,
    t.week_no,
    COUNT(DISTINCT t.basket_id)                                        AS n_baskets,
    COUNT(*)                                                           AS n_items,
    COUNT(DISTINCT p.department)                                       AS n_departments,
    SUM(t.sales_value)                                                 AS net_sales,
    -SUM(t.retail_disc + t.coupon_disc + t.coupon_match_disc)          AS discount_amount,
    SUM(CASE WHEN t.coupon_disc < 0 THEN 1 ELSE 0 END)                 AS n_coupon_items
FROM transaction_data AS t
LEFT JOIN product AS p ON p.product_id = t.product_id
GROUP BY t.household_key, t.week_no
ORDER BY t.household_key, t.week_no
