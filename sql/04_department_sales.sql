-- Sales by product department: join the transaction lines to the product table.
-- Step 1 (CTE): total sales per department. Step 2: divide by the grand total, computed with a window SUM over all rows.

WITH department_totals AS (
    SELECT
        p.department,
        COUNT(DISTINCT t.household_key) AS n_households,
        SUM(t.sales_value)              AS sales
    FROM transaction_data AS t
    INNER JOIN product AS p ON p.product_id = t.product_id
    GROUP BY p.department
)
SELECT
    department,
    n_households,
    sales,
    sales / SUM(sales) OVER () AS share_of_sales
FROM department_totals
ORDER BY sales DESC
