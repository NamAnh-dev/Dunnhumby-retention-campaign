-- Gaps between consecutive purchase weeks of each household, with the window function LAG.
-- silent_weeks = 0 means the household bought again the very next week.
-- Used in notebook 01 to choose the churn definition.

WITH purchase_weeks AS (
    SELECT household_key, week_no
    FROM transaction_data
    GROUP BY household_key, week_no
    HAVING SUM(sales_value) > 0
),
with_previous AS (
    SELECT
        household_key,
        week_no,
        LAG(week_no) OVER (PARTITION BY household_key ORDER BY week_no) AS previous_week
    FROM purchase_weeks
)
SELECT
    household_key,
    previous_week,
    week_no,
    week_no - previous_week - 1 AS silent_weeks
FROM with_previous
WHERE previous_week IS NOT NULL      -- the first purchase of a household has no previous week
ORDER BY household_key, week_no
