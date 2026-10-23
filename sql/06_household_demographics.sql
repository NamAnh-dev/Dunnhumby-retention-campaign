-- One row per household (all 2,500), with cleaned demographics.
-- Only ~1/3 of households have demographics, so the LEFT JOIN leaves NULLs: they are handled in the model pipeline.
-- Cleaning done here:
--   * text ranges -> numbers (income "50-74K" -> 62, household size "5+" -> 5) so the model can use their order
--   * "Unknown" / "None/Unknown" -> NULL, so a missing value looks the same everywhere

WITH households AS (
    SELECT DISTINCT household_key FROM transaction_data
)
SELECT
    h.household_key,
    CASE WHEN d.household_key IS NULL THEN 0 ELSE 1 END AS has_demographics,
    d.age_desc AS age_group,
    CASE d.income_desc
        WHEN 'Under 15K' THEN 10   WHEN '15-24K'   THEN 20  WHEN '25-34K'   THEN 30
        WHEN '35-49K'    THEN 42   WHEN '50-74K'   THEN 62  WHEN '75-99K'   THEN 87
        WHEN '100-124K'  THEN 112  WHEN '125-149K' THEN 137 WHEN '150-174K' THEN 162
        WHEN '175-199K'  THEN 187  WHEN '200-249K' THEN 225 WHEN '250K+'    THEN 275
    END AS income_k,
    CASE WHEN d.household_size_desc = '5+' THEN 5
         ELSE CAST(d.household_size_desc AS INTEGER)
    END AS household_size,
    CASE WHEN d.homeowner_desc IN ('Homeowner', 'Probable Owner') THEN 'owner'
         WHEN d.homeowner_desc IN ('Renter', 'Probable Renter')   THEN 'renter'
    END AS homeowner
FROM households AS h
LEFT JOIN hh_demographic AS d ON d.household_key = h.household_key
ORDER BY h.household_key
