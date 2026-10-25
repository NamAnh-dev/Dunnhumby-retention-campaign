-- RFM segmentation over the whole period (descriptive only: it uses future data, so it is NOT a model feature).
-- recency   = weeks since the last active week (lower is better)
-- frequency = number of active weeks
-- monetary  = total net sales
-- Segments are defined by R and F (behaviour); M (sales) is used afterwards to check how much each segment is worth.
--
-- R score: fixed week thresholds, because about half of the households bought in the very last week, so 5 equal-sized
-- groups (NTILE) would give households with the same recency different scores. 8+ silent weeks = the churn definition.
--   0 weeks -> 5,  1 -> 4,  2-3 -> 3,  4-7 -> 2,  8+ -> 1
-- F score: 5 equal-sized groups with NTILE (5 = most active weeks). Frequency has ~100 different values, so few
-- households tie; household_key breaks those ties, so a household gets the same group on every run.

WITH household_week AS (
    SELECT household_key, week_no, SUM(sales_value) AS net_sales
    FROM transaction_data
    GROUP BY household_key, week_no
    HAVING SUM(sales_value) > 0
),
rfm AS (
    SELECT
        household_key,
        (SELECT MAX(week_no) FROM transaction_data) - MAX(week_no) AS recency_weeks,
        COUNT(*)                                                    AS frequency_weeks,
        SUM(net_sales)                                              AS monetary
    FROM household_week
    GROUP BY household_key
),
scored AS (
    SELECT
        *,
        CASE WHEN recency_weeks = 0 THEN 5
             WHEN recency_weeks = 1 THEN 4
             WHEN recency_weeks <= 3 THEN 3
             WHEN recency_weeks <= 7 THEN 2
             ELSE 1
        END                                                     AS r_score,
        NTILE(5) OVER (ORDER BY frequency_weeks, household_key) AS f_score
    FROM rfm
)
SELECT
    *,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 THEN 'champions'
        WHEN r_score <= 2 AND f_score >= 4 THEN 'at_risk_loyal'
        WHEN r_score >= 4 AND f_score <= 2 THEN 'new_or_light'
        WHEN r_score <= 2 AND f_score <= 2 THEN 'lost'
        ELSE 'middle'
    END AS segment
FROM scored
ORDER BY household_key
