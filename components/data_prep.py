import os
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, size, lit, array, row_number, array, max as spark_max
import numpy as np
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType

print("imported")

def extract_tc(year, spark):
    '''
    Helper function for **TC_trip_formulation**

    Parameters:
        @year: year of the trip data (int)

    Returns:
        Pyspark Dataframe: Tier Earning History by session from RCX
    '''

    mgm_reward = spark.read.parquet('dbfs:/mnt/proddatalake/prod/CFA/CFA_overall_lvl')
    TC = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_{year}.parquet')
    TC = TC.join(mgm_reward, F.col('playerid') == F.col('MLifeID'), how='inner').select(*TC.columns, 'guest_id')

    return TC

def TC_trip_formulation(year, trip, spark):
    '''
    Linking the trips with Tier Earning History for each customer in the given year

    Parameters:
        @year: year of the trip data (int):

        @param trip: trip data
            Trip filtered by different conditions

    Returns:
        Pyspark Dataframe: Trip level tier credits earning
    '''

    customer_demo = spark.read.parquet('/mnt/proddatalake/prod/CFA/CFA_Trip/')\
        .where('TripStart between "2020-01-01" and "2025-12-31"').select('guest_id','TripStart','TripEnd','gender_cd','age','Location','TripLodgingStatus','tenure_visit_corporate')

    TC = extract_tc(year, spark)
    trip_year = trip.where(f"TripStart >= '{year}-01-01' and TripStart < '{year+1}-01-01'")

    joined_df = trip_year.alias("trip").join(
        TC.alias("tc"),
        (F.col("trip.guest_id") == F.col("tc.guest_id")) & 
        (F.to_date("tc.transactiondate").between(F.col("trip.tripstart"), F.col("trip.tripend"))),
        how="inner"
    )

    aggregate_df = joined_df.groupBy('tc.guest_id','playerid','TripID','TripStart', 'TripEnd','property_name').agg(F.sum('tc.tiercredit').alias('TotalTierCredit'), \
        F.max('TripstartMlifetier').alias('tier'),\
        F.sum(F.when(F.col("lob_rollup") == "Gaming", F.col("tiercredit")).otherwise(0)).alias("gaming_tc"), \
        F.sum(F.when(F.col("lob_rollup") == "Non-Gaming", F.col("tiercredit")).otherwise(0)).alias("non_gaming_tc")
    )
    print(aggregate_df.columns, "agg")
    demo_aggregate_df = aggregate_df.join(customer_demo, on=['guest_id','tripstart','tripend'], how='inner')

    # RCX Tier promotion history
    Tier_History = spark.read.parquet('dbfs:/mnt/proddatalake/dev/RCX/Tier_History.parquet')
    Tier_History = Tier_History.withColumn('tier_before', F.when(F.col('tier_before_change') == 'Sapphire', F.lit(1)).when(F.col('tier_before_change') == 'Pearl', F.lit(2)).when(F.col('tier_before_change') == 'Gold', F.lit(3)).when(F.col('tier_before_change') == 'Platinum', F.lit(4)).otherwise(5)) \
        .withColumn('tier_after', F.when(F.col('tier_after_change') == 'Sapphire', F.lit(1)).when(F.col('tier_after_change') == 'Pearl', F.lit(2)).when(F.col('tier_after_change') == 'Gold', F.lit(3)).when(F.col('tier_after_change') == 'Platinum', F.lit(4)).otherwise(5))

    # Select the tier promotion histories
    Tier_promote = Tier_History.where(f"tier_before < tier_after and tier_after != 5 and change_assigned between '{year}-01-01' and '{year}-12-31'").select('playerid','tier_before','tier_after','change_assigned','mt_reason', 'mt_subreason','mth_reason','mth_subreason')

    # # Get the previous tier change date or the begining of the year
    # window_spec = Window.partitionBy("playerid").orderBy("change_assigned")

    Tier_promote = Tier_promote.withColumn(
        "prev_date",
        F.lit(f"{year}-01-01")
    )

    # Join with the trip level data
    agg_promote = demo_aggregate_df.join(Tier_promote, (aggregate_df["playerid"] == Tier_promote["playerid"]) &
        (F.col('tripstart').between(F.col("prev_date"), F.col("change_assigned"))),
        "left").select(*demo_aggregate_df, 'tier_before','tier_after','change_assigned','prev_date','mt_reason', 'mt_subreason','mth_reason','mth_subreason').drop(Tier_promote.)
    
    print(agg_promote.columns)

    agg_promote = agg_promote.withColumn('calendar_year', F.year(F.col('TripStart')))
    agg_promote = agg_promote.dropna(subset = ['age','gender_cd'])

    return agg_promote


def TC_trip_formulation_daily_model(year, trip, spark):
    '''
    Linking the trips with Tier Earning History for each customer in the given year for daily model
    '''
    customer_demo = spark.read.parquet('/mnt/proddatalake/prod/CFA/CFA_Trip/')\
        .where('TripStart between "2020-01-01" and "2025-12-31"').select('guest_id','TripStart','TripEnd','gender_cd','age','Location','TripLodgingStatus','tenure_visit_corporate')

    TC = extract_tc(year, spark)
    trip_year = trip.where(f"TripStart >= '{year}-01-01' and TripStart < '{year+1}-01-01'")

    joined_df = trip_year.alias("trip").join(
        TC.alias("tc"),
        (F.col("trip.guest_id") == F.col("tc.guest_id")) & 
        (F.to_date("tc.transactiondate").between(F.col("trip.tripstart"), F.col("trip.tripend"))),
        how="inner"
    )

    aggregate_df = joined_df.groupBy('tc.guest_id','playerid','TripID','TripStart', 'TripEnd','property_name').agg(F.sum('tc.tiercredit').alias('TotalTierCredit'), \
        F.max('TripstartMlifetier').alias('tier'),\
        F.sum(F.when(F.col("lob_rollup") == "Gaming", F.col("tiercredit")).otherwise(0)).alias("gaming_tc"), \
        F.sum(F.when(F.col("lob_rollup") == "Non-Gaming", F.col("tiercredit")).otherwise(0)).alias("non_gaming_tc")
    )

    demo_aggregate_df = aggregate_df.join(customer_demo, on=['guest_id','tripstart','tripend'], how='inner')

    Tier_History = spark.read.parquet('dbfs:/mnt/proddatalake/dev/RCX/Tier_History.parquet')
    Tier_History = Tier_History.withColumn('tier_before', F.when(F.col('tier_before_change') == 'Sapphire', F.lit(1)).when(F.col('tier_before_change') == 'Pearl', F.lit(2)).when(F.col('tier_before_change') == 'Gold', F.lit(3)).when(F.col('tier_before_change') == 'Platinum', F.lit(4)).otherwise(5)) \
        .withColumn('tier_after', F.when(F.col('tier_after_change') == 'Sapphire', F.lit(1)).when(F.col('tier_after_change') == 'Pearl', F.lit(2)).when(F.col('tier_after_change') == 'Gold', F.lit(3)).when(F.col('tier_after_change') == 'Platinum', F.lit(4)).otherwise(5))

    # Select the tier promotion histories
    Tier_promote = Tier_History.where(f"tier_before < tier_after and tier_after != 5 and change_assigned between '{year}-01-01' and '{year}-12-31'").select('playerid','tier_before','tier_after','change_assigned','mt_reason', 'mt_subreason','mth_reason','mth_subreason')

    # Get the previous tier change date or the begining of the year
    window_spec = Window.partitionBy("playerid").orderBy("change_assigned")

    Tier_promote = Tier_promote.withColumn(
        "prev_date",
        F.coalesce(F.lag('change_assigned').over(window_spec), F.lit(f"{year}-01-01"))
    )

    # Join with the trip level data
    agg_promote = demo_aggregate_df.join(Tier_promote, (aggregate_df["playerid"] == Tier_promote["playerid"]) &
        (F.col('tripstart').between(F.col("prev_date"), F.col("change_assigned"))),
        "left").select(*demo_aggregate_df, 'tier_before','tier_after','change_assigned','prev_date','mt_reason', 'mt_subreason','mth_reason','mth_subreason')
    

    agg_promote = agg_promote.withColumn('calendar_year', F.year(F.col('TripStart')))
    agg_promote = agg_promote.dropna(subset = ['age','gender_cd'])

    return agg_promote

# Helper function for **success_unsuccess_combination**
# @TC_customer_hist: dataframe with tier credit history and trip level data
# @mgm_reward: dataframe with mgm playerid
# @year: year of the trip data
# @spark: spark session
# @percentage_thresh: threshold for the percentage of tier credit earned to be treated as a natural promoted player
# @return: (dataframe, dataframe)
def special_promo_exclution(TC_customer_hist, mgm_reward, rc, year, spark, percentage_thresh):
    # rc is the table for the description of the special promotion code
    la = spark.read.format("delta").load("dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_loyalty_activities/CurrentState")
    la_set_tier = la.where((F.col('activity_type') == 'Set Tier') & (F.year('created_utc_ts') == year))

    promotion_eval = TC_customer_hist.where(TC_customer_hist.change_assigned.isNotNull()).withColumn('tc_percentage', TC_customer_hist['totaltiercredit'] / TC_customer_hist['target_tc'])\
    .join(rc, TC_customer_hist.reason_code == rc.enum_value, how='left').select(*TC_customer_hist.columns, F.col('label').alias('first_label'), 'tc_percentage')\
        .join(rc, TC_customer_hist.subreason_code == rc.enum_value, how='left').select(*TC_customer_hist.columns, 'first_label',F.col('label').alias('second_label'),'tc_percentage')\
            .join(rc, TC_customer_hist.mth_reason_code == rc.enum_value, how='left').select(*TC_customer_hist.columns, 'first_label', 'second_label', F.col('label').alias('third_label'),'tc_percentage')\
                .join(rc, TC_customer_hist.mth_subreason_code == rc.enum_value, how='left').select(*TC_customer_hist.columns, 'first_label', 'second_label', 'third_label', F.col('label').alias('fourth_label'),'tc_percentage').select(*TC_customer_hist.columns, 'first_label', 'second_label', 'third_label', 'fourth_label','tc_percentage')

    promotion_eval_2 = promotion_eval.where(f"(first_label == 'Tier Evaluation' or first_label is null) and (second_label == 'Tier Evaluation' or second_label is null) and (third_label == 'Tier Evaluation' or third_label is null) and (fourth_label == 'Tier Evaluation' or fourth_label is null) and (cast(change_assigned as date) != '{year}-02-01')").join(mgm_reward, on='guest_id', how='left').select(*promotion_eval.columns, 'mlifeid')
    promotion_eval_2.createOrReplaceGlobalTempView('promotion')

    promotion_eval_3 = promotion_eval_2.join(la_set_tier, (promotion_eval_2.mlifeid == la_set_tier.loyalty_id) & (promotion_eval_2.change_assigned == la_set_tier.created_utc_ts), how='leftanti').where(f'tc_percentage > {percentage_thresh}').select(*promotion_eval_2.columns)

    return promotion_eval, promotion_eval_3

# @TC_customer_hist: dataframe with tier credit history and trip level data
# @mgm_reward: dataframe with mgm playerid
# @year: year of the trip data
# @percentage_thresh: threshold for the percentage of tier credit earned to be treated as a natural promoted player
# @spark: spark session
def success_unsuccess_combination(TC_customer_hist, rc, mgm_reward, year, percentage_threshold, spark):
    promotion_initial, promotion_eval_final = special_promo_exclution(TC_customer_hist, mgm_reward, rc, year, spark, percentage_threshold)

    hist_success_promo = promotion_eval_final.groupby('guest_id').agg((F.max('trip_tier')+1).alias('highest_tier'), 
                                                                      F.max('gaming_tc').alias('gtc'),
                                                                      F.max('non_gaming_tc').alias('ngtc'), 
                                                                      F.max('totaltiercredit').alias('TotalTierCredit'),
                                                                      F.max('trip_num').alias('trip_num'),
                                                                      F.max('trip_days').alias('trip_days'),  
                                                                      F.max('earliest_trip').alias('earliest_trip'),
                                                                      F.max('lodger_trip').alias('lodger_trip'), 
                                                                      F.max('local_trip').alias('local_trip'),
                                                                      )

    hist_unsucess_promo = TC_customer_hist.where(TC_customer_hist.change_assigned.isNull()).groupby('guest_id').agg(F.max('trip_tier').alias('highest_tier'),
                                                                                          (F.max('gaming_tc')).alias('gtc'),
                                                                                          (F.max('non_gaming_tc')).alias('ngtc'), 
                                                                                          F.max('totaltiercredit').alias('TotalTierCredit'),
                                                                                          F.max('trip_num').alias('trip_num'),
                                                                                          F.max('trip_days').alias('trip_days'), 
                                                                                          F.max('earliest_trip').alias('earliest_trip'), 
                                                                                          F.max('lodger_trip').alias('lodger_trip'),
                                                                                          F.max('local_trip').alias('local_trip')
                                                                                          )

    hist_combined = hist_success_promo.union(hist_unsucess_promo).groupby('guest_id').agg(F.max('highest_tier').alias('highest_trip_tier'), 
                                                                    (F.sum('gtc')/F.sum('TotalTierCredit')).alias('gtc_percentage'), 
                                                                    (F.sum('ngtc')/F.sum('TotalTierCredit')).alias('ngtc_percentage'), 
                                                                    F.sum('TotalTierCredit').alias('total_tc'), 
                                                                    F.sum('trip_num').alias('trip_num'),
                                                                    F.sum('trip_days').alias('trip_days'), 
                                                                    F.max('earliest_trip').alias('earliest_trip'),
                                                                    (F.sum('lodger_trip')/F.sum('trip_num')).alias('lodger_percentage'),
                                                                    (F.sum('local_trip')/F.sum('trip_num')).alias('local_percentage')
                                                                    )
    
    hist_special_promotion = promotion_initial.join(hist_success_promo, on='guest_id', how='leftanti')

    return hist_combined, hist_special_promotion

def success_trained(train, year, rc, la, mgm_reward):
    '''
    Get the Label and special exclusion

    Parameters:
        @train: dataframe with tier credit history and trip level data

        @mgm_reward: data frame with mgm playerid

        @year: year of the trip data

        @rc: RCX special code reference

        @la: RCX loyalty activites

    Returns:
        (pyspark df, pyspark df):
            First one is the dataframe with the positive cases
            second one contains the player ids which have special promotion
    '''
    sucess_promotion = train.where(train.change_assigned.isNotNull()).withColumn('tc_percentage', train['totaltiercredit'] / train['target_tc'])\
    .join(rc, train.reason_code == rc.enum_value, how='left').select(*train.columns, F.col('label').alias('first_label'), 'tc_percentage')\
        .join(rc, train.subreason_code == rc.enum_value, how='left').select(*train.columns, 'first_label',F.col('label').alias('second_label'),'tc_percentage')\
            .join(rc, train.mth_reason_code == rc.enum_value, how='left').select(*train.columns, 'first_label', 'second_label', F.col('label').alias('third_label'),'tc_percentage')\
                .join(rc, train.mth_subreason_code == rc.enum_value, how='left').select(*train.columns, 'first_label', 'second_label', 'third_label', F.col('label').alias('fourth_label'),'tc_percentage').select(*train.columns, 'first_label', 'second_label', 'third_label', 'fourth_label','tc_percentage')

    sucess_promotion_natural = sucess_promotion.where(f"(first_label == 'Tier Evaluation' or first_label is null) and (second_label == 'Tier Evaluation' or second_label is null) and (third_label == 'Tier Evaluation' or third_label is null) and (fourth_label == 'Tier Evaluation' or fourth_label is null) and (cast(change_assigned as date) != '{year}-02-01')").join(mgm_reward, on='guest_id', how='left').select(*sucess_promotion.columns, 'mlifeid')

    la_set_tier = la.where((F.col('activity_type') == 'Set Tier') & (F.year('created_utc_ts') == year))

    sucess_promotion_final = sucess_promotion_natural.join(la_set_tier, (sucess_promotion_natural.mlifeid == la_set_tier.loyalty_id), how='leftanti').where('tc_percentage > 0.8').select(*sucess_promotion_natural.columns)

    special_promotion = sucess_promotion.join(sucess_promotion_final, on = 'guest_id', how='leftanti')

    return sucess_promotion_final, special_promotion

def first_trip(TC_trip_2023, hist_combined, hist_special_promotion, special_promotion):
    train = TC_trip_2023.where(TC_trip_2023.row_number == 1).withColumn('target_tc',F.when(F.col('TotalTierCredit')>200000,-1)\
                                                            .when(F.col('TotalTierCredit')>75000,200000)\
                                                            .when(F.col('TotalTierCredit')>20000,75000)\
                                                            .otherwise(20000)).join(special_promotion, on='guest_id', how='leftanti').join(hist_special_promotion, on='guest_id', how='leftanti')\
                                                            .select('guest_id', F.date_diff(F.lit('2023-12-31'), 'TripStart').alias('remaining_days'),
                                                                    'totalTiercredit', 'gaming_tc','non_gaming_tc', 'gender_cd','age', 'location','target_tc')
                                                            
    train = train.join(hist_combined, on='guest_id', how = 'inner')

    return train

def second_trip(TC_trip_2023, hist_combined, hist_special_promotion, special_promotion):
    promotion_with_one = TC_trip_2023.groupBy('guest_id','change_assigned').agg(F.max('row_number').alias('trips_to_promo')).where('trips_to_promo <= 2 and change_assigned is not null').select('guest_id')
    train = TC_trip_2023.where(TC_trip_2023.row_number <= 2).groupBy('guest_id')\
    .agg(F.max('TripStart').alias('TripStart'),
         F.sum('TotalTierCredit').alias('TotalTierCredit'),
         F.sum('gaming_tc').alias('gaming_tc'),
         F.sum('non_gaming_tc').alias('non_gaming_tc'),
         F.max('age').alias('age'),
         F.max('gender_cd').alias('gender_cd'),
         F.max('location').alias('location')).select('guest_id', F.date_diff(F.lit('2023-12-31'), 'TripStart').alias('remaining_days'),'totalTiercredit', 'gaming_tc','non_gaming_tc', 'gender_cd','age', 'location')

    train = train.join(promotion_with_one, on='guest_id', how='leftanti')                                                           
    train = train.join(hist_combined, on='guest_id', how = 'inner')
    train = train.withColumn('target_tc',F.when(F.col('highest_trip_tier') == 1, 20000)\
                                            .when(F.col('highest_trip_tier') == 2, 75000)\
                                                .when(F.col('highest_trip_tier') == 3, 200000).otherwise(-1)).where('target_tc > 0')

    return train

def all_trip(TC_trip_2023, hist_combined, hist_special_promotion, special_promotion):
    row_spec = Window.partitionBy('guest_id').orderBy("TripStart")
    cuml_spec = Window.partitionBy('guest_id').orderBy("TripStart").rangeBetween(Window.unboundedPreceding, 0)
    train = TC_trip_2023.withColumn("rn", F.row_number().over(row_spec))\
        .withColumn("Cuml_tc", F.sum('TotalTierCredit').over(cuml_spec))\
            .withColumn("remaining_days_2023", F.datediff(F.lit('2023-12-31'), 'TripStart')).select('guest_id',"rn", 'cuml_tc','remaining_days_2023','age','gender_cd','tripStart')
    train = train.join(special_promotion, on='guest_id', how='leftanti').join(hist_special_promotion, on='guest_id', how='leftanti').join(hist_combined, on='guest_id', how = 'inner')
    train = train.withColumnRenamed('guest_id', 'train_guest_id').withColumnRenamed('trip_num', 'trip_num_2022')
    return train

def get_trained(train_2023, TC_trip_2023, hist_combined, hist_special_promotion, rc, mgm_reward,la, trained_trip_num):
    sucess_promotion_final, special_promotion = success_trained(train_2023, rc, la, mgm_reward)
    train = None
    if trained_trip_num == 1:
        train = first_trip(TC_trip_2023, hist_combined, hist_special_promotion, special_promotion)
    elif trained_trip_num == 2:
        train = second_trip(TC_trip_2023, hist_combined, hist_special_promotion, special_promotion)
    else:
        print("not supported")
        return
    
    promotion_spec = Window.partitionBy('guest_id').orderBy("change_assigned")
    # Exclude the player who has promoted on their first trip
    sucess_label = sucess_promotion_final.where(f'trip_num > {trained_trip_num}').withColumn("row_number", F.row_number().over(promotion_spec)).where('row_number = 1')

    train = train.join(sucess_label, on='guest_id', how = 'left').withColumn('label', F.when(F.col('change_assigned').isNull(), 0).otherwise(1)).select(
    'guest_id',
    'remaining_days',
    train.totalTiercredit,
    train.gaming_tc,
    train.non_gaming_tc,
    'gender_cd',
    'age',
    'location',
    F.col('local_percentage').alias('local_percentage_2022'),
    F.col('lodger_percentage').alias('lodger_percentage_2022'),
    train.target_tc,
    F.col('highest_trip_tier').alias('highest_trip_tier_2022'),
    F.col('gtc_percentage').alias('gtc_percentage_2022'),
    F.col('ngtc_percentage').alias('ngtc_percentage_2022'),
    F.col('total_tc').alias('total_tc_2022'),
    F.col('earliest_trip').alias('earliest_trip_2022'),
    train.trip_num.alias('trip_num_2022'),
    sucess_label.trip_num.alias('success_trip_num'),
    'label'
    )

    return train

def get_trained_all(train_2023, TC_trip_2023, hist_combined, hist_special_promotion, rc, mgm_reward,la):
    success_promotion_final, special_promotion = success_trained(train_2023, rc, la, mgm_reward)
    train = all_trip(TC_trip_2023, hist_combined, hist_special_promotion, special_promotion)
    
    success_promotion_final = success_promotion_final.withColumn('prev_date', F.coalesce(F.lag('change_assigned').over(Window.partitionBy('guest_id').orderBy('change_assigned')), F.lit('2023-01-01')))

    train = train.join(success_promotion_final, 
               (train.train_guest_id == success_promotion_final.guest_id) & ((F.col('tripStart')>= F.col('prev_date')) & (F.col('tripStart') <= F.col('change_assigned'))), 
               how='left'
               ).select(*train.columns, 'pro_tier', 'change_assigned', 'prev_date')
    
    row_spec = Window.partitionBy('train_guest_id','change_assigned').orderBy(F.desc("TripStart"))
    train = train.withColumn('trip_to_promo', F.row_number().over(row_spec))
    train = train.withColumn('label', F.when(F.col('change_assigned').isNotNull(), F.col('trip_to_promo')).otherwise(F.lit(-1))).drop('trip_to_promo')
    train = train.withColumn('curr_tier', F.coalesce(F.col('pro_tier'), F.col('highest_trip_tier'))).drop('pro_tier','highest_trip_tier')
    train = train.withColumn('target_tc', F.when(F.col('curr_tier') == 1, 20000).when(F.col('curr_tier')==2, 75000).when(F.col('curr_tier')==3, 200000).otherwise(0))
    train = train.withColumn('label', F.col('label')-1).where('label != 0')
    train = train.where('total_tc < target_tc') 
    return train


def get_train_daily(TC_curr, hist_combined, hist_special_promotion, year, TC_trip_curr, train_curr, mgm_reward, rc, la):
    """
    Main function for the training data of daily based model.

    Parameters:
        @TC_curr (pyspark dataframe): 
            A dataframe contains each tier credit earning history:
                - Playerid: guest_id
                - Transactiondate: Datetime of the tier credit earning
                - amount: TC earning amount
                ...
        @hist_combined (pyspark dataframe): 
            A summarized dataframe contains trip and tier credits for each customer in the pervious year     
        @hist_sepcial_promotion (pyspark dataframe):
            A dataframe contains all of the playids which have special promotion in the previous year
        ...

    Returns:
        Pyspark Dataframe: Prepared day level traning data for each customer
            - Each customer has x rows of data, where x is the number of trip days in the year
    """
    # Get natural promotion and sepcial promotion for the current year
    success_promotion_final, special_promotion = success_trained(train_curr, year, rc, la, mgm_reward)
    success_promotion_final = success_promotion_final.withColumn('prev_date', F.coalesce(F.lag('change_assigned').over(Window.partitionBy('guest_id').orderBy('change_assigned')), F.lit(f'{year}-01-01'))).select('guest_id', 'prev_date', 'change_assigned')

    TC_curr = TC_curr.join(mgm_reward, F.col('playerid') == F.col('MLifeID'), how='inner').select(*TC_curr.columns, 'guest_id')
    TC_curr = TC_curr.groupby('guest_id', F.to_date('transactiondate').alias('transactiondate')).agg(F.sum('tiercredit').alias('tiercredit')).withColumn('remaining_days', F.date_diff(F.lit(f'{year}-12-31'), 'transactiondate'))
    
    # Exclude players with any historical special promotion
    TC = TC_curr.join(hist_combined.select('guest_id').distinct(), on='guest_id', how='inner')\
        .join(special_promotion.select('guest_id').distinct(), on='guest_id', how='leftanti')\
            .join(hist_special_promotion.select('guest_id').distinct(), on='guest_id', how='leftanti')


    guest_spec = Window.partitionBy('guest_id').orderBy("transactiondate")
    TC = TC.withColumn('days_num', F.row_number().over(guest_spec)).withColumn('cuml_tc', F.sum('tiercredit').over(guest_spec.rangeBetween(Window.unboundedPreceding, 0)))

    # Get the trip length and trip day number for each trip
    result = TC.alias("tc").join(
        TC_trip_curr.alias("trip"),
        (F.col("tc.guest_id") == F.col("trip.guest_id")) & (F.col("tc.transactiondate").between(F.col("trip.TripStart"), F.col("trip.TripEnd"))),
        how="inner"
        ).select(*[F.col(f"tc.{col}") for col in TC.columns],
        (F.date_diff(F.col("trip.TripEnd"), F.col("trip.TripStart"))+1).alias("trip_duration"),
        (F.date_diff(F.col("tc.transactiondate"), F.col("trip.TripStart"))+1).alias("trip_day"),
        F.coalesce('tier_before', F.substring("tier", 2, 2).cast('int')).alias('trip_tier'),
        F.col("trip.TripLodgingStatus")
        ).join(hist_combined, on='guest_id', how='inner').withColumnRenamed('guest_id', 'train_guest_id')

    result = result.join(success_promotion_final, 
                         (F.col('train_guest_id') == F.col('guest_id')) & (F.col('transactiondate').between(F.col('prev_date'), F.col('change_assigned'))), 
                         how='left').withColumn('label', F.when(F.col('change_assigned').isNotNull(), 1).otherwise(0))
    
    # Assigned Target TC based on trip start tier
    result = result.withColumn('target_tc',F.when(F.col('trip_tier') == 1, 20000)\
                                            .when(F.col('trip_tier') == 2, 75000)\
                                                .when(F.col('trip_tier') == 3, 200000).otherwise(-1)).where('target_tc > 0').withColumn('TripLodgingStatus',  F.substring(F.col("TripLodgingStatus"), 2, 2).cast('int'))
    
    # Assigned label based on the tier promotion date
    result = result.withColumn('label', F.when(F.col('transactiondate') > F.date_add(F.col("change_assigned"), -7), 1).otherwise(0))

    return result

# Get training data for new customers
def get_train_daily_new(TC_curr, year, TC_trip_curr, train_curr, mgm_reward, rc, la):
    """
    Main function for the training data of daily based model.

    Parameters:
        @TC_curr (pyspark dataframe): 
            A dataframe contains each tier credit earning history:
                - Playerid: guest_id
                - Transactiondate: Datetime of the tier credit earning
                - amount: TC earning amount
                ...
        @hist_combined (pyspark dataframe): 
            A summarized dataframe contains trip and tier credits for each customer in the pervious year     
        @hist_sepcial_promotion (pyspark dataframe):
            A dataframe contains all of the playids which have special promotion in the previous year
        ...

    Returns:
        Pyspark Dataframe: Prepared day level traning data for each customer
            - Each customer has x rows of data, where x is the number of trip days in the year
    """
    # Get natural promotion and sepcial promotion for the current year
    success_promotion_final, special_promotion = success_trained(train_curr, year, rc, la, mgm_reward)
    success_promotion_final = success_promotion_final.withColumn('prev_date', F.coalesce(F.lag('change_assigned').over(Window.partitionBy('guest_id').orderBy('change_assigned')), F.lit(f'{year}-01-01'))).select('guest_id', 'prev_date', 'change_assigned')

    TC_curr = TC_curr.join(mgm_reward, F.col('playerid') == F.col('MLifeID'), how='inner').select(*TC_curr.columns, 'guest_id')
    TC_curr = TC_curr.groupby('guest_id', F.to_date('transactiondate').alias('transactiondate')).agg(F.sum('tiercredit').alias('tiercredit')).withColumn('remaining_days', F.date_diff(F.lit(f'{year}-12-31'), 'transactiondate'))

    # Exclude players with any historical special promotion
    TC = TC_curr.join(special_promotion.select('guest_id').distinct(), on='guest_id', how='leftanti')

    guest_spec = Window.partitionBy('guest_id').orderBy("transactiondate")
    TC = TC.withColumn('days_num', F.row_number().over(guest_spec)).withColumn('cuml_tc', F.sum('tiercredit').over(guest_spec.rangeBetween(Window.unboundedPreceding, 0)))

    # Get the trip length and trip day number for each trip
    result = TC.alias("tc").join(
        TC_trip_curr.alias("trip"),
        (F.col("tc.guest_id") == F.col("trip.guest_id")) & (F.col("tc.transactiondate").between(F.col("trip.TripStart"), F.col("trip.TripEnd"))),
        how="inner"
        ).drop("trip.guest_id").select(*[F.col(f"tc.{col}") for col in TC.columns],
        (F.date_diff(F.col("trip.TripEnd"), F.col("trip.TripStart"))+1).alias("trip_duration"),
        (F.date_diff(F.col("tc.transactiondate"), F.col("trip.TripStart"))+1).alias("trip_day"),
        F.coalesce('tier_before', F.substring("tier", 2, 2).cast('int')).alias('trip_tier'),
        F.col("trip.TripLodgingStatus")
        ).withColumnRenamed('guest_id', 'train_guest_id')

    result = result.join(success_promotion_final, 
                         (F.col('train_guest_id') == F.col('guest_id')) & (F.col('transactiondate').between(F.col('prev_date'), F.col('change_assigned'))), 
                         how='left').withColumn('label', F.when(F.col('change_assigned').isNotNull(), 1).otherwise(0))
    
    # Assigned Target TC based on trip start tier
    result = result.withColumn('target_tc',F.when(F.col('trip_tier') == 1, 20000)\
                                            .when(F.col('trip_tier') == 2, 75000)\
                                                .when(F.col('trip_tier') == 3, 200000).otherwise(-1)).where('target_tc > 0').withColumn('TripLodgingStatus',  F.substring(F.col("TripLodgingStatus"), 2, 2).cast('int'))
    
    # Assigned label based on the tier promotion date
    result = result.withColumn('label', F.when(F.col('transactiondate') > F.date_add(F.col("change_assigned"), -7), 1).otherwise(0))

    return result
    





