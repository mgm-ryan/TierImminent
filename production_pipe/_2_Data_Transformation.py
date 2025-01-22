# Databricks notebook source
# MAGIC %run "/Workspace/Users/609399@mgmresorts.com/Tier_Imminent/components/helper.py"
# MAGIC %run "/Workspace/Users/609399@mgmresorts.com/Tier_Imminent/components/data_prep.py"

# COMMAND ----------



# COMMAND ----------

property_name = "Borgata"
trip_source = "data_science_mart.tierimminent_raw." + f"{property_name}"+"_trip_return"
trip = spark.read.table(trip_source)

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

hist_combined_2023, hist_special_promotion_2023 = success_unsuccess_combination(TC_customer_hist_2023, rc, mgm_reward,2023, 0.8, spark)

# COMMAND ----------

TC_trip_formulation(2023, trip, spark).display()

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
    F.max('mth_subreason').alias('mth_subreason_code')
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

# COMMAND ----------

scaler_model_path = "/mnt/proddatalake/dev/TierImminent/scalar/minmax_2024_v1"
features = ['tiercredit', 'remaining_days', 'days_num', 'cuml_tc', 'trip_duration','trip_day','TripLodgingStatus', 'highest_trip_tier', 'gtc_percentage', 'ngtc_percentage', 'total_tc', 'trip_num', 'trip_days', 'earliest_trip', 'lodger_percentage', 'local_percentage', 'target_tc']
train = train_minmax(train, features, scaler_model_path)
train.write.mode("overwrite").parquet('/mnt/proddatalake/dev/TierImminent/data/training_data_2024_v1.parquet')
t = spark.read.parquet('dbfs:/mnt/proddatalake/dev/TierImminent/data/training_data_2024_v1.parquet')

unsucess_id = t.groupby('train_guest_id').agg(F.count('change_assigned').alias('count')).where('count = 0').sample(withReplacement=False, fraction=0.5, seed=269)
train_final = t.join(unsucess_id, on = 'train_guest_id', how='leftanti').select(*train.columns)
train_final = train_final.where('(change_assigned != transactiondate) or change_assigned is null')