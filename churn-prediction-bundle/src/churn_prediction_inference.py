# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction Inference
# MAGIC Load the trained LightGBM model from MLflow and score all current subscribers for churn risk.
# MAGIC
# MAGIC **Model:** `classic_hweaver_catalog.gold.churn_prediction`
# MAGIC **Source data:** `classic_hweaver_catalog.silver.users`, `classic_hweaver_catalog.silver.viewing_sessions`, `classic_hweaver_catalog.silver.user_current_subscription`

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install mlflow lightgbm --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load trained model from MLflow
import mlflow
import pandas as pd
import numpy as np

# ── Load the latest model from MLflow experiment ─────────────────────────────
experiment_path = "/Users/haley.weaver@databricks.com/churn_prediction"
experiment = mlflow.get_experiment_by_name(experiment_path)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1,
)
latest_run_id = runs.iloc[0]["run_id"]
optimal_threshold = float(runs.iloc[0]["params.optimal_threshold"])

model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
print(f"Loaded model from run {latest_run_id}")
print(f"Optimal threshold: {optimal_threshold}")

# COMMAND ----------

# DBTITLE 1,Build feature matrix from silver tables
# ── Load data ────────────────────────────────────────────────────────────────
users_df = spark.table("classic_hweaver_catalog.silver.users").toPandas()
subs_df  = spark.table("classic_hweaver_catalog.silver.user_current_subscription").toPandas()
sess_df  = spark.table("classic_hweaver_catalog.silver.viewing_sessions").toPandas()

# ── Aggregate viewing sessions per user ──────────────────────────────────────
user_sessions = sess_df.groupby("user_id").agg(
    total_sessions     = ("session_id", "count"),
    total_watch_sec    = ("active_watch_time_sec", "sum"),
    avg_completion_pct = ("max_position_pct", "mean"),
    completion_rate    = ("completion_flag", "mean"),
    abandon_rate       = ("abandon_flag", "mean"),
    avg_buffers        = ("num_buffers", "mean"),
    avg_pauses         = ("num_pauses", "mean"),
    distinct_titles    = ("title", "nunique"),
    last_session       = ("session_start_ts", "max"),
).reset_index()

user_sessions["total_watch_hours"] = user_sessions["total_watch_sec"] / 3600
now_ts = pd.Timestamp.now(tz="UTC")
if user_sessions["last_session"].dt.tz is None:
    user_sessions["last_session"] = user_sessions["last_session"].dt.tz_localize("UTC")
user_sessions["days_since_last_session"] = (now_ts - user_sessions["last_session"]).dt.days

# ── Join all tables ──────────────────────────────────────────────────────────
df = (
    users_df.merge(subs_df, on="user_id", how="inner")
            .merge(user_sessions, on="user_id", how="left")
)

# ── Build feature matrix (must match training pipeline) ─────────────────────
categorical_features = ["age_band", "gender", "country", "acquisition_channel"]
numeric_features = [
    "tenure_days", "age", "profile_count", "max_streams",
    "lifetime_events", "total_upgrades", "total_downgrades",
    "total_cancels", "total_reactivations", "payment_failures",
    "total_sessions", "total_watch_hours", "avg_completion_pct",
    "completion_rate", "abandon_rate", "avg_buffers", "avg_pauses",
    "distinct_titles", "days_since_last_session",
]

df_score = df[categorical_features + numeric_features].copy()
df_encoded = pd.get_dummies(df_score, columns=categorical_features, drop_first=False, dtype=int)

# Align columns with training features
train_features = model.feature_names_in_
for col in train_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
X_score = df_encoded[train_features].fillna(0).astype(float)

print(f"Feature matrix: {X_score.shape[0]} users × {X_score.shape[1]} features")

# COMMAND ----------

# DBTITLE 1,Generate churn predictions
# ── Score all users ──────────────────────────────────────────────────────────
df["churn_probability"] = model.predict_proba(X_score)[:, 1]
df["churn_risk"] = pd.cut(
    df["churn_probability"],
    bins=[0, 0.2, optimal_threshold, 0.6, 1.0],
    labels=["Low", "Medium", "High", "Critical"],
)
df["predicted_churn"] = (df["churn_probability"] >= optimal_threshold).astype(int)

# ── Display results ──────────────────────────────────────────────────────────
results = df[[
    "user_id", "current_status", "current_plan", "age_band", "country",
    "tenure_days", "total_cancels", "total_sessions", "avg_completion_pct",
    "churn_probability", "churn_risk", "predicted_churn",
]].sort_values("churn_probability", ascending=False)

print(f"Scored {len(results)} users")
print(f"\nChurn risk distribution:")
print(results["churn_risk"].value_counts().to_string())
print(f"\nPredicted churners: {results['predicted_churn'].sum()} ({results['predicted_churn'].mean():.1%})")
print(f"\n── Top 20 highest churn risk users ──")
display(results.head(20))

# COMMAND ----------

# DBTITLE 1,Save predictions to gold table
from datetime import datetime

# ── Prepare output DataFrame ─────────────────────────────────────────────────
output = results.copy()
output["scored_at"] = datetime.utcnow()
output["model_run_id"] = latest_run_id
output["threshold"] = optimal_threshold

# ── Write to Unity Catalog gold table ────────────────────────────────────────
output_sdf = spark.createDataFrame(output)
(
    output_sdf.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("classic_hweaver_catalog.gold.churn_predictions")
)

print(f"Saved {output_sdf.count()} predictions to classic_hweaver_catalog.gold.churn_predictions")
print(f"Model run: {latest_run_id}")
print(f"Threshold: {optimal_threshold}")
