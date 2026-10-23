-- Period-over-period retention with 4-week periods:
-- of the households active in period p, what share is also active in period p + 1?
-- Uses a self LEFT JOIN: a household with no match in the next period is "not retained".
-- Only complete 4-week periods are used (the data has 102 weeks, so weeks 101-102 are dropped).

WITH period_active AS (
    SELECT DISTINCT
        household_key,
        (week_no - 1) // 4 + 1 AS period
    FROM transaction_data
    WHERE week_no <= (SELECT MAX(week_no) FROM transaction_data) // 4 * 4
    GROUP BY household_key, week_no
    HAVING SUM(sales_value) > 0
)
SELECT
    cur.period,
    COUNT(*)                                   AS active_households,
    COUNT(nxt.household_key)                   AS retained_next_period,
    COUNT(nxt.household_key) * 1.0 / COUNT(*)  AS retention_rate
FROM period_active AS cur
LEFT JOIN period_active AS nxt
    ON nxt.household_key = cur.household_key
   AND nxt.period = cur.period + 1
WHERE cur.period < (SELECT MAX(period) FROM period_active)   -- the last period has no "next" yet
GROUP BY cur.period
ORDER BY cur.period
