import os
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, collect_list, size, lit, array, row_number, array, max as spark_max
import numpy as np
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
import mlflow

min_max_scalar_path = '/mnt/proddatalake/dev/TierImminent/scalar/min_max_model'

def create_sequences_and_labels(df, features,sequence_length, padding_value=0.0):
    if df.count() == 0:
        return [], [], [], []
    
    df = df.withColumn("features", F.array(*features))
    
    # grouped_df = df.groupBy('train_guest_id').agg(
    #     collect_list("features").alias("all_trips"),
    #     collect_list('label').alias("all_labels")
    # )

    grouped_df = df.groupBy('train_guest_id').agg(
        F.sort_array(collect_list(F.struct("days_num", "features")), asc=True).alias("all_trips"),
        F.sort_array(collect_list(F.struct("days_num", "label")), asc=True).alias("all_labels")
    )

    grouped_df = grouped_df.withColumn(
        "all_trips", F.expr("transform(all_trips, x -> x.features)")
    ).withColumn(
        "all_labels", F.expr("transform(all_labels, x -> x.label)")
    )

    def create_padded_sequences_and_labels(trip_list, label_list, ids):
        if not trip_list:
            return [], [],[],[]
        num_features = len(trip_list[0])
        sequences, labels, length, train_id_ls = [], [], [] ,[]
        for i in range(0, len(trip_list), sequence_length):
            sequence = trip_list[i:i + sequence_length]
            label = label_list[i:i + sequence_length]
            labels.append(float(label[-1]))
            length.append(float(len(sequence)))
            if len(sequence) < sequence_length:
                sequence += [[padding_value] * num_features] * (sequence_length - len(sequence))
                label += [padding_value] * (sequence_length - len(label))
            sequences.append(sequence)
            #labels.append(float(max(label)))
            train_id_ls.append(int(ids))
        return sequences, labels, length, train_id_ls

    schema = StructType([
        StructField("sequences", ArrayType(ArrayType(ArrayType(FloatType()))), True),
        StructField("labels", ArrayType(FloatType()), True),
        StructField("length", ArrayType(FloatType()), True),
        StructField("id_ls", ArrayType(IntegerType()), True)
    ])
    create_sequences_udf = udf(lambda trips, labels, ids: create_padded_sequences_and_labels(trips, labels, ids), schema)

    result_df = grouped_df.withColumn(
        "sequences_and_labels", create_sequences_udf(col("all_trips"), col("all_labels"), col("train_guest_id"))
    )
    result_df = result_df.select(
        col('train_guest_id'),
        col("sequences_and_labels.sequences").alias("sequences"),
        col("sequences_and_labels.labels").alias("labels"),
        col("sequences_and_labels.length").alias('length'),
        col("sequences_and_labels.id_ls").alias("id_ls")
    )
    exploded_df = result_df.withColumn("sequence", col("sequences")).withColumn("label", col("labels")).withColumn("length", col('length'))
    def to_numpy_array(iterator):
        for row in iterator:
            for seq, lbl, ln, ids in zip(row.sequences, row.labels, row.length, row.id_ls):
                yield (np.array(seq, dtype=float), lbl, ln, ids)

    numpy_rdd = exploded_df.rdd.mapPartitions(to_numpy_array)
    numpy_data = list(numpy_rdd.collect())

    sequences, labels, length, ids = zip(*numpy_data)
    sequences = np.stack(sequences, axis=0)
    labels = np.array(labels, dtype=float)
    length = np.array(length, dtype=float)
    ids = np.array(ids, dtype=int)
    # exploded_df = (
    #     exploded_df
    #     .select(
    #         F.posexplode(col("sequence")).alias("index", "sequence"),  # Explode with position and value
    #         col("label"),
    #         col("length"),
    #         col("id_ls")
    #     )
    # )

    # aligned_df = (
    #     exploded_df
    #     .withColumn("label", col("label")[col("index")])  # Align labels based on index
    #     .withColumn("length", col("length")[col("index")])  # Align lengths based on index
    #     .withColumn("id", col("id_ls")[col("index")])  # Align ids based on index
    #     .select("sequence", "label", "length", "id")  # Keep only necessary columns
    # )

    # numpy_data = aligned_df.collect()

    # sequences = np.array([row['sequence'] for row in numpy_data], dtype=float)
    # labels = np.array([row['label'] for row in numpy_data], dtype=float)
    # length = np.array([row['length'] for row in numpy_data], dtype=float)
    # ids = np.array([row['id'] for row in numpy_data], dtype=int)

    return sequences, labels, length, ids

def train_minmax(df, features):
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    assembled_df = assembler.transform(df)
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

    scaler_model = scaler.fit(assembled_df)
    scaled_assembled_df = scaler_model.transform(assembled_df)
    #scaler_model.write().overwrite().save(min_max_scalar_path)
    #print(f"Scalar Model saved at the path: {min_max_scalar_path}")
    scaled_assembled_df = scaled_assembled_df.withColumn("features_array", vector_to_array(col("scaledFeatures")))
    # Unpack
    for i, col_name in enumerate(features):
        scaled_assembled_df = scaled_assembled_df.withColumn(f"{col_name}_scaled", col("features_array").getItem(i))

    scaled_assembled_df = scaled_assembled_df.drop("features").drop("scaledFeatures")
    return scaler_model, scaled_assembled_df

def apply_minmax(df, features, min_max_scalar_path):
    loaded_scaler_model = mlflow.spark.load_model(min_max_scalar_path)
    print(f"Scalar Model loaded from path: {min_max_scalar_path}")
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    test_data = assembler.transform(df)
    scaled_test_data = loaded_scaler_model.transform(test_data)
    scaled_test_data = scaled_test_data.drop('features')

    scaled_test_data = scaled_test_data.withColumn("features_array", vector_to_array(col("scaledFeatures")))
    # Unpack
    for i, col_name in enumerate(features):
        scaled_test_data = scaled_test_data.withColumn(f"{col_name}_scaled", col("features_array").getItem(i))

    scaled_test_data = scaled_test_data.drop("features").drop("scaledFeatures")
    
    return scaled_test_data











    