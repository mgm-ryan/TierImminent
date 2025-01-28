import os
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, size, lit, array, row_number, array, max as spark_max
import numpy as np
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType
from components.helper import *
from components.data_prep import *
import torch
from model.nn import * 
import pandas as pd
from datetime import datetime

from databricks.sdk.runtime import *

def get_return_trip_borgata(mgm_reward):
    cdp_path='/dbfs/mnt/cdpprod/Customer_Profile_Aggregates/'
    yesterday=str(max(os.listdir(cdp_path)))
    trip_data = spark.read.parquet('dbfs:/mnt/cdpprod/Customer_Profile_Aggregates/'+yesterday+'/')\
        .withColumn('Site', F.when(F.col('SiteGroup') == 'LAS', 'Vegas').otherwise('Region'))\
            .filter(F.col('Mnth')>='2016-01-01')\
                .filter((F.col('TripRvMgmt_Segment')!='Convention')|F.col('TripRvMgmt_Segment').isNull())
    trip_data = trip_data.withColumn("TripID",F.concat(F.col("Guest_ID"),F.lit('_'), F.col("TripStart"),F.lit('_'),F.col("TripEnd")))

    CPA = trip_data.where("property_name != 'BetMGM' and tripstart <= '2025-12-31'").select('guest_id','Property_Name','Department','TripStart','TripEnd','TripStartMlifeTier', 'TripGamingDays','TripID')
    trip_CPA = CPA.groupBy('guest_id','TripStart','TripEnd', 'TripID').agg(F.count('Department').alias('dept_num'), F.max('Property_Name').alias('property_name'),
                                                                                F.max('TripGamingDays').alias('TripGamingDays'), F.max('TripStartMlifeTier').alias('TripStartMlifeTier'))
    trip_borgata = trip_CPA.where(F.col('property_name').contains("Borgata"))
    #trip_borgata_2023 = trip_borgata.where('TripStart between "2023-01-01" and "2023-12-31"')
    trip_borgata_2024 = trip_borgata.where('TripStart between "2024-01-01" and "2024-12-31"')
    trip_borgata_2025 = trip_borgata.where('TripStart between "2025-01-01" and "2025-12-31"')
    trip_spec = trip_borgata_2024.join(trip_borgata_2025, on='guest_id', how='inner').select('guest_id').distinct()

    # Tier Credits Earning History from RCX
    #tc_2023 = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_2023.parquet')
    tc_2024 = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_2024.parquet')
    tc_2025 = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_2024.parquet')
    tc_combined = tc_2024.union(tc_2025)

    tc_combined = tc_combined.groupBy('playerid').agg(
        F.sum('tiercredit').alias('total_tc'), 
        F.sum(F.when(F.col('site_name').contains('Borgata'), F.col('tiercredit')).otherwise(F.lit(0))).alias('borgata_tc')
    )

    tc_combined = tc_combined.join(mgm_reward, F.col('playerid') == F.col('MLifeID'), how='inner').select(*tc_combined.columns, 'guest_id')

    # Filtering Borgata Dominant Players
    borgata_dominant_players = tc_combined.where('borgata_tc > (total_tc * 0.8)').select('guest_id')
    # Filtering Trip Data
    trip = trip_CPA.join(trip_spec, on='guest_id', how='inner').join(borgata_dominant_players, on='guest_id', how='inner')

    #write_log(f'[{datetime.utcnow()}] STEP [2] - Trips for retruning and Borgata Dominant players created')
    return trip

def hist(trip, rc, mgm_reward, year, percentage_threshold, spark):
    TC_trip_prev = TC_trip_formulation(year, trip, spark)
    TC_customer_hist = TC_trip_prev.groupby('guest_id','calendar_year','change_assigned').agg(
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
        'earliest_trip', F.date_diff(F.lit(f'{year}-12-31'),'earliest_trip')
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

    return success_unsuccess_combination(TC_customer_hist, rc, mgm_reward, year, percentage_threshold, spark)


def get_new_customer(mgm_reward):
    cdp_path='/dbfs/mnt/cdpprod/Customer_Profile_Aggregates/'
    yesterday=str(max(os.listdir(cdp_path)))
    trip_data = spark.read.parquet('dbfs:/mnt/cdpprod/Customer_Profile_Aggregates/'+yesterday+'/')\
        .withColumn('Site', F.when(F.col('SiteGroup') == 'LAS', 'Vegas').otherwise('Region'))\
            .filter(F.col('Mnth')>='2016-01-01')\
                .filter((F.col('TripRvMgmt_Segment')!='Convention')|F.col('TripRvMgmt_Segment').isNull())
    trip_data = trip_data.withColumn("TripID",F.concat(F.col("Guest_ID"),F.lit('_'), F.col("TripStart"),F.lit('_'),F.col("TripEnd")))

    CPA = trip_data.where("property_name != 'BetMGM' and tripstart < '2025-12-31'").select('guest_id','Property_Name','Department','TripStart','TripEnd','TripStartMlifeTier', 'TripGamingDays','TripID')
    trip_CPA = CPA.groupBy('guest_id','TripStart','TripEnd', 'TripID').agg(F.count('Department').alias('dept_num'), F.max('Property_Name').alias('property_name'),
                                                                                F.max('TripGamingDays').alias('TripGamingDays'), F.max('TripStartMlifeTier').alias('TripStartMlifeTier'))
    trip_borgata = trip_CPA.where(F.col('property_name').contains("Borgata"))
    trip_borgata_2024 = trip_borgata.where('TripStart between "2024-01-01" and "2024-12-31"')
    trip_borgata_2025 = trip_borgata.where('TripStart between "2025-01-01" and "2025-12-31"')
    trip_spec = trip_borgata_2024.join(trip_borgata_2024, on='guest_id', how='inner').select('guest_id').distinct()

    tc_2025 = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_2025.parquet')
    tc_2025 = tc_2025.groupBy('playerid').agg(
        F.sum('tiercredit').alias('total_tc'), 
        F.sum(F.when(F.col('site_name').contains('Borgata'), F.col('tiercredit')).otherwise(F.lit(0))).alias('borgata_tc')
    )

    tc_2025 = tc_2025.join(mgm_reward, F.col('playerid') == F.col('MLifeID'), how='inner').select(*tc_2025.columns, 'guest_id')

    # Filtering Borgata Dominant Players
    borgata_dominant_players = tc_2025.where('borgata_tc > 0').select('guest_id')
    # Filtering Trip Data
    trip = trip_borgata_2025.join(trip_spec, on='guest_id', how='leftanti').join(borgata_dominant_players, on='guest_id', how='inner')

    return trip

               
def get_inference(TC_curr, hist_combined, hist_special_promotion, year, TC_trip_curr, mgm_reward):
    TC_curr = TC_curr.groupby('guest_id', F.to_date('transactiondate').alias('transactiondate')).agg(F.sum('tiercredit').alias('tiercredit')).withColumn('remaining_days', F.date_diff(F.lit(f'{year}-12-31'), 'transactiondate')) 

    guest_spec = Window.partitionBy('guest_id').orderBy("transactiondate")
    TC = TC_curr.withColumn('days_num', F.row_number().over(guest_spec)).withColumn('cuml_tc', F.sum('tiercredit').over(guest_spec.rangeBetween(Window.unboundedPreceding, 0)))

    # Get the trip length and trip day number for each trip
    result = TC.alias("tc").join(
        TC_trip_curr.alias("trip"),
        (F.col("tc.guest_id") == F.col("trip.guest_id")) & (F.col("tc.transactiondate").between(F.col("trip.TripStart"), F.col("trip.TripEnd"))),
        how="inner"
        ).select(*[F.col(f"tc.{col}") for col in TC.columns],
        (F.date_diff(F.col("trip.TripEnd"), F.col("trip.TripStart"))+1).alias("trip_duration"),
        (F.date_diff(F.col("tc.transactiondate"), F.col("trip.TripStart"))+1).alias("trip_day"),
        F.substring(F.col("trip.tier"), 2, 2).cast('int').alias('trip_tier'),
        F.col("trip.TripLodgingStatus")
        ).join(hist_combined, on='guest_id', how='inner').withColumnRenamed('guest_id', 'train_guest_id')

    # Assigned Target TC based on trip start tier
    result = result.withColumn('target_tc',F.when(F.col('trip_tier') == 1, 20000)\
                                            .when(F.col('trip_tier') == 2, 75000)\
                                                .when(F.col('trip_tier') == 3, 200000).otherwise(-1)).where('target_tc > 0').withColumn('TripLodgingStatus',  F.substring(F.col("TripLodgingStatus"), 2, 2).cast('int'))

    return result

def get_inference_new(year, TC_trip_curr, mgm_reward):
    TC = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_2025.parquet')
    TC = TC.join(mgm_reward, F.col('playerid') == F.col('MLifeID'), how='inner').select(*TC.columns, 'guest_id')
    TC = TC.groupby('guest_id', F.to_date('transactiondate').alias('transactiondate')).agg(F.sum('tiercredit').alias('tiercredit')).withColumn('remaining_days', F.date_diff(F.lit(f'{year}-12-31'), 'transactiondate'))

    guest_spec = Window.partitionBy('guest_id').orderBy("transactiondate")
    TC = TC.withColumn('days_num', F.row_number().over(guest_spec)).withColumn('cuml_tc', F.sum('tiercredit').over(guest_spec.rangeBetween(Window.unboundedPreceding, 0)))

    # Get the trip length and trip day number for each trip
    result = TC.alias("tc").join(
        TC_trip_curr.alias("trip"),
        (F.col("tc.guest_id") == F.col("trip.guest_id")) & (F.col("tc.transactiondate").between(F.col("trip.TripStart"), F.col("trip.TripEnd"))),
        how="inner"
        ).drop("trip.guest_id").select(*[F.col(f"tc.{col}") for col in TC.columns], 'trip_tier',
        (F.date_diff(F.col("trip.TripEnd"), F.col("trip.TripStart"))+1).alias("trip_duration"),
        (F.date_diff(F.col("tc.transactiondate"), F.col("trip.TripStart"))+1).alias("trip_day"),
        F.col("trip.TripLodgingStatus")
        ).withColumnRenamed('guest_id', 'train_guest_id')

    result = result.withColumn('target_tc',F.when(F.col('trip_tier') == 1, 20000)\
                                            .when(F.col('trip_tier') == 2, 75000)\
                                                .when(F.col('trip_tier') == 3, 200000).otherwise(-1)).where('target_tc > 0').withColumn('TripLodgingStatus',  F.substring(F.col("TripLodgingStatus"), 2, 2).cast('int'))
    
    return result

def inference_formulation(trip, TC, hist_combined_prev, hist_special_promotion_prev, year, mgm_reward, returning_customer = False):
    customer_demo = spark.read.parquet('/mnt/proddatalake/prod/CFA/CFA_Trip/')\
        .where(f'TripStart >= "{year}-01-01"').select('guest_id','TripStart','TripEnd','TripLodgingStatus')

    if returning_customer:
        trip_year = trip.where(f"TripStart >= '{year}-01-01' ")

        joined_df = trip_year.alias("trip").join(
            TC.alias("tc"),
            (trip_year.guest_id == TC.guest_id) & 
            (F.to_date("tc.transactiondate").between(F.col("trip.tripstart"), F.col("trip.tripend"))),
            how="inner"
        )

        aggregate_df = joined_df.groupBy('tc.guest_id','playerid','TripID','TripStart', 'TripEnd','property_name').agg(F.sum('tc.tiercredit').alias('TotalTierCredit'), \
            F.max('TripstartMlifetier').alias('tier'),\
            F.sum(F.when(F.col("lob_rollup") == "Gaming", F.col("tiercredit")).otherwise(0)).alias("gaming_tc"), \
            F.sum(F.when(F.col("lob_rollup") == "Non-Gaming", F.col("tiercredit")).otherwise(0)).alias("non_gaming_tc")
        )

        demo_aggregate_df = aggregate_df.join(customer_demo, on=['guest_id','tripstart','tripend'], how='inner')

        inference_data = get_inference(TC, hist_combined_prev, hist_special_promotion_prev, 2025, demo_aggregate_df, mgm_reward)

        return inference_data
    
    else:
        joined_df = trip.join(
            TC.alias("tc"),
            (trip.guest_id == TC.guest_id) & 
            (F.to_date("tc.transactiondate").between(F.col("tripstart"), F.col("tripend"))),
            how="inner"
        )

        aggregate_df = joined_df.groupBy('tc.guest_id','playerid','TripID','TripStart', 'TripEnd','property_name').agg(F.sum('tc.tiercredit').alias('TotalTierCredit'), \
            F.max('TripstartMlifetier').alias('tier'),\
            F.sum(F.when(F.col("lob_rollup") == "Gaming", F.col("tiercredit")).otherwise(0)).alias("gaming_tc"), \
            F.sum(F.when(F.col("lob_rollup") == "Non-Gaming", F.col("tiercredit")).otherwise(0)).alias("non_gaming_tc")
        )

        demo_aggregate_df = aggregate_df.join(customer_demo, on=['guest_id','tripstart','tripend'], how='inner').\
            withColumn('trip_tier', F.substring("tier", 2, 2).cast('int')).\
                withColumn('target_TC', F.when(F.col('trip_tier')==1, 20000).when(F.col('trip_tier')==2, 75000).when(F.col('trip_tier')==3, 200000).otherwise(-1)).\
                    filter('target_tc != -1')

        inference_data = get_inference_new(2025, demo_aggregate_df, mgm_reward)
        return inference_data




def inference_driver():
    mgm_reward = spark.read.parquet('dbfs:/mnt/proddatalake/prod/CFA/CFA_overall_lvl')
    rc = spark.read.format("delta").load('dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_enums/CurrentState')

    year = 2025
    TC = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_{year}.parquet')
    TC = TC.join(mgm_reward, F.col('playerid') == F.col('MLifeID'), how='inner').select(*TC.columns, 'guest_id')
    
    # returning customers
    #----------------------------------------------------------
    trip = get_return_trip_borgata(mgm_reward)

    hist_combined_prev, hist_special_promotion_prev = hist(trip, rc, mgm_reward, 2024, 0.8, spark)

    inference_data = inference_formulation(trip, TC, hist_combined_prev, hist_special_promotion_prev, year, mgm_reward, returning_customer = True)

    # Get yesterday customers
    cus_list = inference_data.where("date(transactiondate) = date_sub(current_date(), 1)").select('train_guest_id')
    cus_df = inference_data.join(cus_list, (inference_data.train_guest_id == cus_list.train_guest_id), how = 'inner').drop(inference_data.train_guest_id)
    cus_df = cus_df.withColumn('rn', F.row_number().over(Window.partitionBy('train_guest_id').orderBy('remaining_days'))).where('rn <= 6')
    # #-----------------------------------------------------------

    # # new customers
    # #-----------------------------------------------------------
    new_trip = get_new_customer(mgm_reward)
    inference_data_new = inference_formulation(new_trip, TC, None, None, year, mgm_reward, returning_customer = False)
    cus_new_list = inference_data_new.where("date(transactiondate) = date_sub(current_date(), 1)").select('train_guest_id')
    new_cus_df = inference_data_new.join(cus_new_list, (inference_data_new.train_guest_id == cus_new_list.train_guest_id), how = 'inner').drop(inference_data_new.train_guest_id)
    new_cus_df = new_cus_df.withColumn('rn', F.row_number().over(Window.partitionBy('train_guest_id').orderBy('remaining_days'))).where('rn <= 6')

    return inference_data, inference_data_new

def get_inference_sequence(df, features, sequence_length, padding_value=0.0):
    df = df.withColumn("features", F.array(*features))
    
    window = Window.partitionBy('train_guest_id').orderBy('days_num')
    df = df.withColumn("row_number", row_number().over(window))
    
    grouped_df = df.groupBy('train_guest_id').agg(
        collect_list("features").alias("all_trips")
    )

    def create_padded_sequences_and_labels(trip_list, ids):
        if not trip_list:
            return [], [],[],[]
        num_features = len(trip_list[0])
        sequences, length, train_id_ls = [], [] ,[]
        for i in range(0, len(trip_list), sequence_length):
            sequence = trip_list[i:i + sequence_length]
            length.append(float(len(sequence)))
            if len(sequence) < sequence_length:
                sequence += [[padding_value] * num_features] * (sequence_length - len(sequence))
            sequences.append(sequence)
            train_id_ls.append(float(ids))
        return sequences, length, train_id_ls

    schema = StructType([
        StructField("sequences", ArrayType(ArrayType(ArrayType(FloatType()))), True),
        StructField("length", ArrayType(FloatType()), True),
        StructField("id_ls", ArrayType(FloatType()), True)
    ])
    create_sequences_udf = udf(lambda trips, ids: create_padded_sequences_and_labels(trips, ids), schema)

    result_df = grouped_df.withColumn(
        "sequences_and_labels", create_sequences_udf(col("all_trips"), col("train_guest_id"))
    )
    result_df = result_df.select(
        col('train_guest_id'),
        col("sequences_and_labels.sequences").alias("sequences"),
        col("sequences_and_labels.length").alias('length'),
        col("sequences_and_labels.id_ls").alias("id_ls")
    )
    exploded_df = result_df.withColumn("sequence", col("sequences")).withColumn("length", col('length'))
    def to_numpy_array(iterator):
        for row in iterator:
            for seq, ln, ids in zip(row.sequences, row.length, row.id_ls):
                yield (np.array(seq, dtype=float), ln, ids)

    numpy_rdd = exploded_df.rdd.mapPartitions(to_numpy_array)
    numpy_data = list(numpy_rdd.collect())

    sequences, length, ids = zip(*numpy_data)
    sequences = np.stack(sequences, axis=0)
    length = np.array(length, dtype=float)
    ids = np.array(ids, dtype=float)

    return sequences, length, ids

def prediction_return(df, threshold):
    features = ['tiercredit', 'remaining_days', 'days_num', 'cuml_tc', 'trip_duration','trip_day','TripLodgingStatus', 'highest_trip_tier', 'gtc_percentage', 'ngtc_percentage', 'total_tc', 'trip_num', 'trip_days', 'earliest_trip', 'lodger_percentage', 'local_percentage', 'target_tc']
    scaled_df = apply_minmax(df, features)
    
    scaled_df = scaled_df.withColumn("features_array", vector_to_array(col("scaledFeatures")))
    
    # Unpack
    for i, col_name in enumerate(features):
        scaled_df = scaled_df.withColumn(f"{col_name}_scaled", col("features_array").getItem(i))


    test_np, length, train_id = get_inference_sequence(scaled_df, features, 6)

    #load model
    model = torch.load('/Workspace/Users/609399@mgmresorts.com/Tier_Imminent/artifacts/lstm.pth', map_location=torch.device('cpu'))

    X = torch.tensor(test_np, dtype=torch.float32).cpu()
    len_X = torch.tensor(length, dtype=torch.float32).cpu()
    with torch.no_grad():
        outputs = model(X, len_X).squeeze()
    outputs = torch.sigmoid(outputs)

    y_pred = [1 if i >= threshold else 0 for i in outputs]
    final_pred = pd.DataFrame({'guest_id': train_id, 'pred':y_pred})

    write_log(f'[{datetime.utcnow()}] STEP [3] - Customers selected for inference has been generated, number of customers:{len(final_pred)}')

    return final_pred

    


# if __name__ == "__main__":
#     # inf = inference_driver()
#     # test = get_test_2024()

#     # test.where('train_guest_id = 4739845').display()
#     # inf.where('train_guest_id = 4739845').display()
#     # -------------------------------------------------------
#     # df = inference_driver()
#     # out = prediction(df, 0.35) 
#     # out.to_csv(f'/Workspace/Users/609399@mgmresorts.com/Tier_Imminent/result/{datetime.now().strftime("%Y-%m-%d")}.csv', index=False)
#     # --------------------------------------------------------
#     inference_driver()












