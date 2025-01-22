# Databricks notebook source
import pandas as pd
import pyspark.sql.functions as F

cdp_path='dbfs:/mnt/cdpprod/Customer_Profile_Aggregates/'
mgm_reward = spark.read.parquet('dbfs:/mnt/proddatalake/prod/CFA/CFA_overall_lvl')
rc = spark.read.format("delta").load('dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_enums/CurrentState')
la = spark.read.format("delta").load("dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_loyalty_activities/CurrentState")

# COMMAND ----------

property_name = 'Borgata'

# COMMAND ----------

files = dbutils.fs.ls(cdp_path)
yesterday = str(max([f.name for f in files]))

trip_data = spark.read.parquet('dbfs:/mnt/cdpprod/Customer_Profile_Aggregates/'+yesterday+'/')\
            .withColumn('Site', F.when(F.col('SiteGroup') == 'LAS', 'Vegas').otherwise('Region'))\
            .filter(F.col('Mnth')>='2016-01-01')\
            .filter((F.col('TripRvMgmt_Segment')!='Convention')|F.col('TripRvMgmt_Segment').isNull())
trip_data = trip_data.withColumn("TripID",F.concat(F.col("Guest_ID"),F.lit('_'), F.col("TripStart"),F.lit('_'),F.col("TripEnd")))

CPA = trip_data.where("property_name != 'BetMGM' and tripstart < '2024-12-31'").select('guest_id','Property_Name','Department','TripStart','TripEnd','TripStartMlifeTier', 'TripGamingDays','TripID')
trip_CPA = CPA.groupBy('guest_id','TripStart','TripEnd', 'TripID').agg(F.count('Department').alias('dept_num'), F.max('Property_Name').alias('property_name'),
                                                                              F.max('TripGamingDays').alias('TripGamingDays'), F.max('TripStartMlifeTier').alias('TripStartMlifeTier'))
trip_property = trip_CPA.where(F.col('property_name').contains(f"{property_name}"))
trip_2023 = trip_property.where('TripStart between "2023-01-01" and "2023-12-31"')
trip_2024 = trip_property.where('TripStart between "2024-01-01" and "2024-12-31"')

trip_spec = trip_2024.join(trip_2023, on='guest_id', how='inner').select('guest_id').distinct()

# COMMAND ----------

# Tier Credits Earning History from RCX
tc_2023 = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_2023.parquet')
tc_2024 = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_2024.parquet')

tc_combined = tc_2023.union(tc_2024)

property_dominant_alias = f"{property_name}"+"_tc"

tc_combined = tc_combined.groupBy('playerid').agg(
    F.sum('tiercredit').alias('total_tc'), 
    F.sum(F.when(F.col('site_name').contains(f'Borgata'), F.col('tiercredit')).otherwise(F.lit(0))).alias(f'{property_dominant_alias}')
)

tc_combined = tc_combined.join(mgm_reward, F.col('playerid') == F.col('MLifeID'), how='inner').select(*tc_combined.columns, 'guest_id')

# Filtering Borgata Dominant Players
borgata_dominant_players = tc_combined.where(f'{property_dominant_alias} > (total_tc * 0.8)').select('guest_id')
# Filtering Trip Data
trip = trip_CPA.join(trip_spec, on='guest_id', how='inner').join(borgata_dominant_players, on='guest_id', how='inner')

# COMMAND ----------

path = "data_science_mart.tierimminent_raw."
table_name = f"{property_name}"+"_trip_return"
trip.write.format("delta").mode("overwrite").saveAsTable(path+table_name)