# Databricks notebook source

# MAGIC %md
# MAGIC # Content Engagement Stats Analysis
# MAGIC This notebook reads from `classic_hweaver_catalog.gold.content_roi_metrics` and
# MAGIC `classic_hweaver_catalog.gold.viewer_journey_funnel`, performs statistical analysis
# MAGIC using **numpy** and **scipy**, and writes results to
# MAGIC `classic_hweaver_catalog.gold.content_engagement_stats`.

# COMMAND ----------

import numpy as np
from scipy import stats
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read source tables
# MAGIC Aggregate `content_roi_metrics` (a metric view) and `viewer_journey_funnel` at the content level.

# COMMAND ----------

# Read content_roi_metrics — metric view with space-delimited column names
roi_df = spark.sql("""
    SELECT
        `Content ID`   AS content_id,
        `Title`        AS title,
        `Content Type` AS content_type,
        `Genre`        AS genre,
        `Is Original`  AS is_original,
        `Total Sessions`          AS total_sessions,
        `Unique Viewers`           AS unique_viewers,
        `Total Watch Hours`        AS total_watch_hours,
        `Avg Watch Time Min`       AS avg_watch_time_min,
        `Completion Rate`          AS completion_rate,
        `Abandon Rate`             AS abandon_rate,
        `Engagement Score`         AS engagement_score,
        `Health Index`             AS health_index,
        `Avg User Rating`          AS avg_user_rating
    FROM classic_hweaver_catalog.gold.content_roi_metrics
""")

print(f"ROI metrics rows: {roi_df.count()}")
roi_df.show(5, truncate=False)

# COMMAND ----------

# Read viewer_journey_funnel — aggregate to content level
funnel_df = spark.sql("""
    SELECT
        content_id,
        SUM(browse_impressions)    AS total_impressions,
        SUM(tile_clicks)           AS total_clicks,
        SUM(play_starts)           AS total_play_starts,
        SUM(completions)           AS total_completions,
        AVG(ctr_browse_to_click)   AS avg_ctr,
        AVG(cvr_click_to_play)     AS avg_cvr,
        AVG(pct_complete)          AS avg_pct_complete,
        AVG(dropoff_click_to_play) AS avg_dropoff_click_to_play,
        AVG(dropoff_25_to_50)      AS avg_dropoff_25_to_50,
        AVG(dropoff_50_to_75)      AS avg_dropoff_50_to_75,
        AVG(dropoff_75_to_complete) AS avg_dropoff_75_to_complete
    FROM classic_hweaver_catalog.gold.viewer_journey_funnel
    GROUP BY content_id
""")

print(f"Funnel rows: {funnel_df.count()}")
funnel_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Join datasets and collect for NumPy / SciPy analysis

# COMMAND ----------

# Inner join on content_id
joined_df = roi_df.join(funnel_df, on="content_id", how="inner")
print(f"Joined rows: {joined_df.count()}")

# Collect to pandas for numpy/scipy operations
pdf = joined_df.toPandas()
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Statistical analysis with NumPy and SciPy
# MAGIC - **Z-scores** on health index and engagement score to flag outliers
# MAGIC - **Pearson correlation** between engagement score and funnel completion rate
# MAGIC - **Spearman correlation** between health index and CTR

# COMMAND ----------

# --- Z-scores for outlier detection ---
pdf["health_index_zscore"] = stats.zscore(pdf["health_index"].fillna(0).values)
pdf["engagement_score_zscore"] = stats.zscore(pdf["engagement_score"].fillna(0).values)

# Flag outliers (|z| > 2)
pdf["health_outlier"] = np.abs(pdf["health_index_zscore"]) > 2
pdf["engagement_outlier"] = np.abs(pdf["engagement_score_zscore"]) > 2

print(f"Health index outliers:     {pdf['health_outlier'].sum()}")
print(f"Engagement score outliers: {pdf['engagement_outlier'].sum()}")

# COMMAND ----------

# --- Pearson correlation: engagement_score vs funnel completion rate ---
mask = pdf["engagement_score"].notna() & pdf["avg_pct_complete"].notna()
pearson_r, pearson_p = stats.pearsonr(
    pdf.loc[mask, "engagement_score"].values,
    pdf.loc[mask, "avg_pct_complete"].values,
)
print(f"Pearson r (engagement vs completion): {pearson_r:.4f}  p-value: {pearson_p:.4e}")

# --- Spearman correlation: health_index vs CTR ---
mask2 = pdf["health_index"].notna() & pdf["avg_ctr"].notna()
spearman_r, spearman_p = stats.spearmanr(
    pdf.loc[mask2, "health_index"].values,
    pdf.loc[mask2, "avg_ctr"].values,
)
print(f"Spearman r (health vs CTR):            {spearman_r:.4f}  p-value: {spearman_p:.4e}")

# COMMAND ----------

# --- Percentile ranks using numpy ---
pdf["engagement_percentile"] = np.round(
    stats.rankdata(pdf["engagement_score"].fillna(0).values, method="average")
    / len(pdf) * 100, 2
)
pdf["health_percentile"] = np.round(
    stats.rankdata(pdf["health_index"].fillna(0).values, method="average")
    / len(pdf) * 100, 2
)

# --- Composite quality score (weighted blend) ---
w_engagement, w_health, w_completion = 0.4, 0.35, 0.25
pdf["composite_quality_score"] = np.round(
    w_engagement * pdf["engagement_percentile"]
    + w_health * pdf["health_percentile"]
    + w_completion * (pdf["avg_pct_complete"].fillna(0) * 100),
    2,
)

pdf[["title", "engagement_percentile", "health_percentile", "composite_quality_score"]].head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write results to `classic_hweaver_catalog.gold.content_engagement_stats`

# COMMAND ----------

# Select final columns
output_cols = [
    "content_id", "title", "content_type", "genre", "is_original",
    # ROI metrics
    "total_sessions", "unique_viewers", "total_watch_hours",
    "avg_watch_time_min", "completion_rate", "abandon_rate",
    "engagement_score", "health_index", "avg_user_rating",
    # Funnel metrics
    "total_impressions", "total_clicks", "total_play_starts",
    "total_completions", "avg_ctr", "avg_cvr", "avg_pct_complete",
    "avg_dropoff_click_to_play", "avg_dropoff_25_to_50",
    "avg_dropoff_50_to_75", "avg_dropoff_75_to_complete",
    # Statistical features
    "health_index_zscore", "engagement_score_zscore",
    "health_outlier", "engagement_outlier",
    "engagement_percentile", "health_percentile",
    "composite_quality_score",
]

output_sdf = spark.createDataFrame(pdf[output_cols])

output_sdf.write.mode("overwrite").saveAsTable(
    "classic_hweaver_catalog.gold.content_engagement_stats"
)

print("✓ Table written: classic_hweaver_catalog.gold.content_engagement_stats")
spark.table("classic_hweaver_catalog.gold.content_engagement_stats").show(5, truncate=False)
