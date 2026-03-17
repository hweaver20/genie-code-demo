# genie-code-demo

Databricks Asset Bundle for the **Content Engagement Stats** ETL pipeline.

## What it does

Reads from two gold-layer tables, performs statistical analysis (z-scores, Pearson/Spearman correlations, percentile ranks, composite scoring) using NumPy and SciPy, and writes enriched results to `classic_hweaver_catalog.gold.content_engagement_stats`.

### Source tables
- `classic_hweaver_catalog.gold.content_roi_metrics`
- `classic_hweaver_catalog.gold.viewer_journey_funnel`

### Output table
- `classic_hweaver_catalog.gold.content_engagement_stats`

## Project structure

```
├── databricks.yml                              # DAB bundle configuration & job definition
└── src/
    └── content_engagement_stats_analysis.py    # Analysis notebook
```

## Deployment

```bash
# Validate the bundle
databricks bundle validate

# Deploy to your target workspace
databricks bundle deploy

# Run the job
databricks bundle run content_engagement_stats_job
```
