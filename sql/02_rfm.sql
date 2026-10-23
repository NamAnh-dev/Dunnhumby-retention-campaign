-- RFM segmentation over the whole period (descriptive only: it uses future data, so it is NOT a model feature).
-- recency  = weeks since the last active week (lower is better)
-- frequency = number of active weeks
-- monetary = total net sales
-- Each is split into 5 equal-sized groups with NTILE (5 = best).
-- Many households have the same value (e.g. recency 0), so household_key breaks the ties:
-- without it, NTILE puts tied households in a different group on every run.

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
        NTILE(5) OVER (ORDER BY recency_weeks DESC, household_key) AS r_score,
        NTILE(5) OVER (ORDER BY frequency_weeks, household_key)    AS f_score,
        NTILE(5) OVER (ORDER BY monetary, household_key)           AS m_score
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
