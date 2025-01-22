import lightgbm as lgb
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

def balanced(train):
    positive_df = train.filter(F.col("label") == 1)
    negative_df = train.filter(F.col("label") == 0)

    positive_count = positive_df.count()
    negative_count = negative_df.count()

    sample_count = min(positive_count, negative_count)

    sampled_positive_df = positive_df.sample(withReplacement=False, fraction=sample_count / positive_count)
    sampled_negative_df = negative_df.sample(withReplacement=False, fraction=(sample_count / negative_count))

    balanced_train_sample = sampled_positive_df.union(sampled_negative_df)
    
    return balanced_train_sample

def preprocess_train(train):
    train = train.dropna(subset=['location']).where('target_tc != -1')
    train_sample = balanced(train)
    indexer_location = StringIndexer(inputCol="location", outputCol="location_index")
    encoder_location = OneHotEncoder(inputCol="location_index", outputCol="location_cat")

    indexer_gender = StringIndexer(inputCol="Gender_cd", outputCol="gender")

    feature_cols = ['remaining_days',
                    'totalTiercredit',
                    'gaming_tc',
                    'non_gaming_tc',
                    'gender',
                    'age',
                    'location_cat',
                    'local_percentage_2022',
                    'lodger_percentage_2022',
                    'target_tc',
                    'highest_trip_tier_2022',
                    'gtc_percentage_2022',
                    'ngtc_percentage_2022',
                    'total_tc_2022',
                    'earliest_trip_2022',
                    'trip_num_2022',
                    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    pipeline = Pipeline(stages=[indexer_location, encoder_location, indexer_gender, assembler])

    preprocess = pipeline.fit(train_sample)
    preprocessed_train = preprocess.transform(train_sample)

    preprocessed_train = preprocessed_train.withColumn('label', F.when(F.col('success_trip_num').isNull(), 0)\
                                                            .when((F.col('success_trip_num')>2) & (F.col('success_trip_num')<=5), 1)\
                                                                .when((F.col('success_trip_num')>5) & (F.col('success_trip_num')<=11), 2)\
                                                                    .when((F.col('success_trip_num')>11) & (F.col('success_trip_num')<=26), 3)\
                                                                        .otherwise(4))
    
    preprocessed_final = preprocessed_train.select('features', 'label')

    return preprocessed_final







