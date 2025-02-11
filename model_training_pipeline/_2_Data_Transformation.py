# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC from mlflow.models import infer_signature
# MAGIC from components.helper import *
# MAGIC from components.data_prep import *
# MAGIC from components.config import *
# MAGIC import mlflow

# COMMAND ----------

property_name = "Borgata"
trip_source = "data_science_mart.tierimminent_raw." + f"{property_name}"+"_trip_return"
trip = spark.read.table(trip_source)

mgm_reward = spark.read.parquet('dbfs:/mnt/proddatalake/prod/CFA/CFA_overall_lvl')
rc = spark.read.format("delta").load('dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_enums/CurrentState')
la = spark.read.format("delta").load("dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_loyalty_activities/CurrentState")

# COMMAND ----------

trip.select(F.countDistinct('guest_id')).show()

# COMMAND ----------

TC_trip_2023 = TC_trip_formulation(2023, trip, spark)
TC_customer_hist_2023 = TC_trip_2023.groupby('guest_id','calendar_year','change_assigned').agg(
    F.count('*').alias('trip_num'),
    F.min('tripstart').alias('earliest_trip'),
    F.sum(F.date_diff('tripend', 'tripstart')+1).alias('trip_days'),
    F.sum('TotalTierCredit').alias('TotalTierCredit'),
    F.sum('gaming_tc').alias('gaming_tc'),
    F.sum('non_gaming_tc').alias('non_gaming_tc'),
    F.sum(F.when(F.col('TripLodgingStatus') == "01 - In House", 1).otherwise(0)).alias('lodger_trip'),
    F.sum(F.when(F.col('TripLodgingStatus') == "02 - Local", 1).otherwise(0)).alias('local_trip'),
    F.sum(F.when(F.col('TripLodgingStatus') == "03 - Drop In", 1).otherwise(0)).alias('non_loager_trip'),
    F.min('tier').alias('tier'),
    F.max('tier_before').alias('pro_tier'),
    F.max('mt_reason').alias('reason_code'),
    F.max('mt_subreason').alias('subreason_code'),
    F.max('mth_reason').alias('mth_reason_code'),
    F.max('mth_subreason').alias('mth_subreason_code')
).withColumn(
    'earliest_trip', F.date_diff(F.lit('2023-12-31'),'earliest_trip')
).withColumn(
    'trip_tier',
    F.coalesce('pro_tier', F.substring("tier", 2, 2).cast('int'))
).withColumn(
    'target_TC', 
    F.when(F.col('trip_tier')==1, 20000)
    .when(F.col('trip_tier')==2, 75000)
    .when(F.col('trip_tier')==3, 200000)
    .otherwise(-1)
).filter('target_tc != -1')

hist_combined_2023, hist_special_promotion_2023 = success_unsuccess_combination(TC_customer_hist_2023, rc, mgm_reward, 2023, 0.8, spark)

# COMMAND ----------

# MAGIC %md
# MAGIC # Training Based on 2024

# COMMAND ----------

TC_trip_2024 = TC_trip_formulation_daily_model(2024, trip, spark)
train_2024 = TC_trip_2024.groupby('guest_id','calendar_year','change_assigned').agg(
    F.count('*').alias('trip_num'), 
    F.sum('TotalTierCredit').alias('TotalTierCredit'),
    F.sum('gaming_tc').alias('gaming_tc'),
    F.sum('non_gaming_tc').alias('non_gaming_tc'),
    F.min('tier').alias('tier'),
    F.max('tier_before').alias('pro_tier'),
    F.max('mt_reason').alias('reason_code'),
    F.max('mt_subreason').alias('subreason_code'),
    F.max('mth_reason').alias('mth_reason_code'),
    F.max('mth_subreason').alias('mth_subreason_code'),
    F.max('tier_after').alias('tier_after')
).withColumn(
    'trip_tier',
    F.coalesce('pro_tier', F.substring("tier", 2, 2).cast('int'))
).withColumn(
    'target_TC', 
    F.when(F.col('trip_tier')==1, 20000)
    .when(F.col('trip_tier')==2, 75000)
    .when(F.col('trip_tier')==3, 200000)
    .otherwise(-1)
).filter('target_tc != -1')

windowSpec = Window.partitionBy("guest_id").orderBy("TripStart")
TC_trip_2024 = TC_trip_2024.withColumn("row_number", F.row_number().over(windowSpec))


# COMMAND ----------

year = 2024
TC = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_{year}.parquet')
#train = get_train_daily(TC, hist, hist_special_promotion, 2024, TC_trip_2024, train_2024, mgm_reward, rc, la)
# 2023 only
train = get_train_daily(TC, hist_combined_2023, hist_special_promotion_2023, 2024, TC_trip_2024, train_2024, mgm_reward, rc, la)
train = train.distinct()

# COMMAND ----------

error= train.where('cuml_tc > target_tc and (transactiondate != change_assigned or change_assigned is null)').select('train_guest_id').distinct()
train = train.join(error, on = 'train_guest_id', how='leftanti')

# COMMAND ----------

train.where('label=1').count()

# COMMAND ----------

features = ['tiercredit', 'remaining_days', 'days_num', 'cuml_tc', 'trip_duration','trip_day','TripLodgingStatus', 'highest_trip_tier', 'gtc_percentage', 'ngtc_percentage', 'total_tc', 'trip_num', 'trip_days', 'earliest_trip', 'lodger_percentage', 'local_percentage', 'target_tc']


minmax_scalar, scaled_data = train_minmax(train, features)

# import mlflow
# mlflow.set_registry_uri("databricks-uc")

# with mlflow.start_run():
#     mlflow.spark.log_model(minmax_scalar, "minmax_scaler_Borgata_return", 
#                            registered_model_name="data_science_mart.tierimminent_cleaned.minmax_scaler_Borgata_return")


# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

signature = infer_signature(train, scaled_data)

with mlflow.start_run():
    mlflow.spark.log_model(minmax_scalar, "minmax_scaler_Borgata_return", signature = signature,
                           registered_model_name="data_science_mart.tierimminent_cleaned.minmax_scaler_Borgata_return")

# COMMAND ----------

path = "data_science_mart.tierimminent_cleaned."
table_name = f"{property_name}"+"_trip_cleaned_return"
scaled_data.write.format("delta").mode("overwrite").saveAsTable(path+table_name)

# unsucess_id = t.groupby('train_guest_id').agg(F.count('change_assigned').alias('count')).where('count = 0').sample(withReplacement=False, fraction=0.5, seed=269)
# train_final = t.join(unsucess_id, on = 'train_guest_id', how='leftanti').select(*train.columns)
# train_final = train_final.where('(change_assigned != transactiondate) or change_assigned is null')

# COMMAND ----------

# MAGIC %md
# MAGIC # New Customers

# COMMAND ----------

new_trip_source = "data_science_mart.tierimminent_raw." + f"{property_name}"+"_trip_new"
new_trip = spark.read.table(new_trip_source)

# COMMAND ----------

temp = TC_trip_formulation_daily_model(2024, new_trip, spark)
temp_train = temp.groupby('guest_id','calendar_year','change_assigned').agg(
    F.count('*').alias('trip_num'), 
    F.sum('TotalTierCredit').alias('TotalTierCredit'),
    F.sum('gaming_tc').alias('gaming_tc'),
    F.sum('non_gaming_tc').alias('non_gaming_tc'),
    F.min('tier').alias('tier'),
    F.max('tier_before').alias('pro_tier'),
    F.max('mt_reason').alias('reason_code'),
    F.max('mt_subreason').alias('subreason_code'),
    F.max('mth_reason').alias('mth_reason_code'),
    F.max('mth_subreason').alias('mth_subreason_code'),
    F.max('tier_after').alias('tier_after')
).withColumn(
    'trip_tier',
    F.coalesce('pro_tier', F.substring("tier", 2, 2).cast('int'))
).withColumn(
    'target_TC', 
    F.when(F.col('trip_tier')==1, 20000)
    .when(F.col('trip_tier')==2, 75000)
    .when(F.col('trip_tier')==3, 200000)
    .otherwise(-1)
).filter('target_tc != -1')

year = 2024
TC = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_{year}.parquet')

new_train_daily = get_train_daily_new(TC, 2024, temp, temp_train, mgm_reward, rc, la)

window_spec = Window.partitionBy('train_guest_id').orderBy('transactiondate').rowsBetween(Window.unboundedPreceding, 0)

new_train_daily = new_train_daily.withColumn(
    "lodger_percentage",
    (F.sum(F.when(F.col('TripLodgingStatus')==1, 1).otherwise(0)).over(window_spec) / F.row_number().over(window_spec))
).withColumn(
    "local_percentage",
    (F.sum(F.when(F.col('TripLodgingStatus')==2, 1).otherwise(0)).over(window_spec) / F.row_number().over(window_spec))
)

new_train_daily = new_train_daily.distinct()

# COMMAND ----------

error= new_train_daily.where('cuml_tc > target_tc and (transactiondate != change_assigned or change_assigned is null)').select('train_guest_id').distinct()
new_train_daily = new_train_daily.join(error, on = 'train_guest_id', how='leftanti')

# COMMAND ----------

new_train_daily.where('label = 1').count()

# COMMAND ----------

minmax_scalar_new, scaled_data_new = train_minmax(new_train_daily, FEATURE_NAMES_NEW)

# COMMAND ----------

signature = infer_signature(new_train_daily, scaled_data_new)
mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run():
    mlflow.spark.log_model(minmax_scalar_new, "minmax_scaler_Borgata_new", signature = signature,
                           registered_model_name="data_science_mart.tierimminent_cleaned.minmax_scaler_Borgata_new")

# COMMAND ----------

path = "data_science_mart.tierimminent_cleaned."
table_name = f"{property_name}"+"_trip_cleaned_new"
scaled_data_new.write.format("delta").mode("overwrite").saveAsTable(path+table_name)
