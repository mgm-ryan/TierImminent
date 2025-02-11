# Databricks notebook source
import pandas as pd
import numpy as np
from delta import *
from pyspark.sql import functions as F
import os
import torch

from model.nn import *
from components.data_prep import *
from components.helper import *
from components.config import *
import matplotlib.pyplot as plt

import shap
#spark.conf.set("spark.sql.caseSensitive", "false")
output_path = '/mnt/proddatalake/dev/RCX/'
%load_ext autoreload
%autoreload 2

# COMMAND ----------

trip_data = spark.read.parquet('dbfs:/mnt/cdpprod/Customer_Profile_Aggregates/2025-02-05')
trip_data = trip_data.groupBy('guest_id','TripStart','TripEnd').agg(F.sum('net_gaming_revenue').alias('win_loss'))

# COMMAND ----------

cust_profile_path = 'dbfs:/mnt/cdpprod/Customer_Profile_Customer_nopii/'+'2025-02-03'+'/';
customer_data = spark.read.parquet(cust_profile_path)

# COMMAND ----------

cfa = spark.read.parquet("dbfs:/mnt/proddatalake/dev/CFA/CFA_overall_lvl_Region_Borgata")

# COMMAND ----------

linked = spark.read.parquet('/mnt/proddatalake/dev/RCX/Individual_Balances_For_LinkedAccounts.parquet')
mgm_reward = spark.read.parquet('dbfs:/mnt/proddatalake/prod/CFA/CFA_overall_lvl')
linked = linked.join(mgm_reward, linked.player_id == mgm_reward.MLifeID, how='inner').select('account_id','guest_id','tiercreditsearned')

# COMMAND ----------

new_cus = spark.read.table("data_science_mart.tierimminent_cleaned.borgata_trip_cleaned_new")
# return_cus.join(linked, return_cus.train_guest_id == linked.guest_id, how='inner').select('account_id','train_guest_id','tiercreditsearned').distinct().groupby('account_id').count().display()

# COMMAND ----------

return_cus.join(linked, return_cus.train_guest_id == linked.guest_id, how='inner').select('account_id','train_guest_id','tiercreditsearned').where('account_id = "63b6bfe1377f2a0027f573e1"').display()

# COMMAND ----------

return_cus = spark.read.table("data_science_mart.tierimminent_cleaned.borgata_trip_cleaned_new")
success_cus = return_cus.groupBy('train_guest_id').agg(F.sum('label').alias('success')).where('success > 0').select('train_guest_id')
return_cus = return_cus.where('transactiondate != change_assigned or change_assigned is null')
return_cus = return_cus.cache()
return_cus.count()
#new_cus = spark.read.parquet("/mnt/proddatalake/dev/TierImminent/data/training_data_2024_v1_newcus.parquet")

# COMMAND ----------

return_cus.where('label = 1').count()

# COMMAND ----------

return_cus.select(*FEATURE_NAMES_RETURN).describe().display()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
loaded_model = mlflow.pytorch.load_model(f"models:/data_science_mart.tierimminent_cleaned.borgata_lstm_model_new@active", map_location=torch.device('cpu'))

# COMMAND ----------

old_model = torch.load('/Workspace/Users/609399@mgmresorts.com/Tier_Imminent/artifacts/model_2024_v1.pth', map_location='cpu', weights_only=False)
old_model.eval()

# COMMAND ----------

from tqdm import tqdm
# outputs_ls, labels_ls, y_pred_ls_2, y_pred_ls_5, y_pred_ls_35 = [], [], [], [], []
df = return_cus
old_outputs_ls = []
new_outputs_ls = []
labels_ls = []
remaining_ls = []
id_ls = []
X, y, l = None, np.array([]), np.array([])
tier = pd.DataFrame(columns=['id','remaining_dyas','tier'])
feature_scaled = []
for j in FEATURE_NAMES_NEW:
    feature_scaled.append(j+'_scaled')
for i in tqdm(range(365, 0, -1)):
    cus_list = df.filter(df.remaining_days == i).select(F.col('train_guest_id').alias('id')).distinct()
    temp_df = df.join(cus_list, (df.train_guest_id == cus_list.id) & (df.remaining_days >= i), how = 'inner')
    temp_df = temp_df.withColumn('rn', F.row_number().over(Window.partitionBy('train_guest_id').orderBy('remaining_days'))).where('rn <= 6')
    # t = temp_df.groupby('train_guest_id','remaining_days').agg(F.min('target_tc').alias('tier')).where(f'remaining_days == {i}')
    # t = t.toPandas()
    # tier = pd.concat([tier, t], ignore_index=True)

    train_np, label, length, train_id = create_sequences_and_labels(temp_df, feature_scaled, 6)
    if X is None:
        X = train_np
    else:
        X = np.append(X, train_np, axis = 0)
    y = np.append(y, label)
    l = np.append(l, length)
    
    test_X = torch.tensor(train_np, dtype=torch.float32).cpu()
    len_X = torch.tensor(length, dtype=torch.float32).cpu()
    with torch.no_grad():
        new_outputs = loaded_model(test_X, len_X).squeeze()

    new_outputs = torch.sigmoid(new_outputs)
    # y_pred_2 = [1 if prob >= 0.2 else 0 for prob in outputs]
    # y_pred_5 = [1 if prob >= 0.5 else 0 for prob in outputs]
    # y_pred_35 = [1 if prob >= 0.35 else 0 for prob in outputs]

    # for i, p in zip(train_id, y_pred):
    #     if p == 1:
    #         if i in output_dict:
    #             output_dict[i] += 1
    #         else:
    #             output_dict[i] = 1
                                                                                                                                                              
    # y_pred_ls_2.extend(y_pred_2)  
    # y_pred_ls_5.extend(y_pred_5)
    # y_pred_ls_35.extend(y_pred_35)
    new_outputs_ls.extend(new_outputs)
    labels_ls.extend(label)
    id_ls.extend(train_id)
    remaining_ls.extend([i]*len(label))


# COMMAND ----------

new_outputs_ls = [tensor.item() for tensor in new_outputs_ls]

# COMMAND ----------

t = pd.DataFrame(columns=['train_guest_id','remaining_days','label'])
t['label'] = labels_ls
t['train_guest_id'] = id_ls
t['remaining_days'] = remaining_ls
t['new_output'] = new_outputs_ls
t['rn'] = t.groupby('train_guest_id')['remaining_days'].transform('count')

t_df = spark.createDataFrame(t)

# COMMAND ----------

analysis_df = return_cus.join(t_df, on=['train_guest_id','remaining_days'], how='inner').join(success_cus, on='train_guest_id', how='leftanti')
above_thresh_cus = analysis_df.groupby('train_guest_id').agg(F.max('new_output').alias('percentage')).where('percentage > 0.5').select('train_guest_id')

# COMMAND ----------

target_analysis = analysis_df.\
    join(above_thresh_cus, on='train_guest_id', how='inner').join(customer_data, analysis_df.train_guest_id == customer_data.guest_id, how='inner').\
        join(cfa, analysis_df.train_guest_id == cfa.guest_id, how='inner').\
            select('train_guest_id',
                    'transactiondate',
                    'tiercredit',
                    'cuml_tc',
                    'trip_duration',
                    'TripLodgingStatus',
                    'target_tc',
                    'new_output',
                    'rn','playertypeltd_corporate', 'l18m_ATW','mlifeenrolleddate')
target_analysis = target_analysis.join(trip_data,
                                       (target_analysis.train_guest_id == trip_data.guest_id) &(target_analysis.transactiondate >= trip_data.TripStart) & (target_analysis.transactiondate <= trip_data.TripEnd), how='inner')

# COMMAND ----------

target_analysis.display()

# COMMAND ----------

target_analysis.groupby('train_guest_id').agg(F.max('playertypeltd_corporate').alias('game')).groupby('game').count().display()

# COMMAND ----------

target_analysis.groupby('TripLodgingStatus').count().display()

# COMMAND ----------

target_analysis.select('train_guest_id').distinct().count()

# COMMAND ----------

re = 362
guest_id = 9056095
#cus_list = return_cus.filter(return_cus.remaining_days == re).select('train_guest_id')
temp_df = return_cus.where((return_cus.train_guest_id == guest_id) & (return_cus.remaining_days >= re))

temp_df = temp_df.withColumn('rn', F.row_number().over(Window.partitionBy('train_guest_id').orderBy('remaining_days'))).where('rn <= 6')
temp_df = temp_df.where(f'train_guest_id = {guest_id}')

train_np, label, length, train_id = create_sequences_and_labels(temp_df, features, 6)

train_np = torch.tensor(train_np, dtype=torch.float32).cpu()
length = torch.tensor(length, dtype=torch.float32).cpu()

with torch.no_grad():
    new_out = loaded_model(train_np, length).squeeze()
    old_out = old_model(train_np, length).squeeze()

new_out = torch.sigmoid(new_out)
old_out = torch.sigmoid(old_out)

features = ['tiercredit', 'remaining_days', 'days_num', 'cuml_tc', 'trip_duration','trip_day','TripLodgingStatus', 'highest_trip_tier', 'gtc_percentage', 'ngtc_percentage', 'total_tc', 'trip_num', 'trip_days', 'earliest_trip', 'lodger_percentage', 'local_percentage', 'target_tc']
temp_df.where(f'train_guest_id = {guest_id}').select('train_guest_id','transactiondate','change_assigned','label', *features, 'features_array').display()

# COMMAND ----------

new_out, old_out

# COMMAND ----------

def cal_metric(thresh, df):
    tpr = len(df[(df['outputs'] > thresh) &(df['label']==1)]) / len(df[df['label']==1])
    fpr = len(df[(df['outputs'] > thresh) & (df['label']==0)]) / len(df[df['label']==0])
    return tpr,fpr, len(df[df['outputs'] > thresh]) / 364

cal_metric(0.3, t)

# COMMAND ----------

def create_test_df():
    train_id = 123
    tiercredit = 5000
    remaining_days = 50
    days_num = 1
    cuml_tc = 5000
    trip_duration = 3
    trip_day = 1
    TripLodgingStatus = 1
    highest_trip_tier = 1
    gtc_percentage = 1
    ngtc_percentage = 0
    total_tc = 0.1234
    trip_num = 1
    trip_days = 0
    earliest_trip = 365
    lodger_percentage = 0
    local_percentage = 1
    target_tc = 20000
    label = 0

    features = ['tiercredit', 'remaining_days', 'days_num', 'cuml_tc', 'trip_duration','trip_day','TripLodgingStatus', 'highest_trip_tier', 'gtc_percentage', 'ngtc_percentage', 'total_tc', 'trip_num', 'trip_days', 'earliest_trip', 'lodger_percentage', 'local_percentage', 'target_tc']

    data = [(train_id, tiercredit, remaining_days, days_num, cuml_tc, trip_duration,trip_day,TripLodgingStatus, highest_trip_tier, gtc_percentage, ngtc_percentage, total_tc, trip_num, trip_days, earliest_trip, lodger_percentage, local_percentage, target_tc, label)]

    columns = ['train_guest_id', 'tiercredit', 'remaining_days', 'days_num', 'cuml_tc', 'trip_duration','trip_day','TripLodgingStatus', 'highest_trip_tier', 'gtc_percentage', 'ngtc_percentage', 'total_tc', 'trip_num', 'trip_days', 'earliest_trip', 'lodger_percentage', 'local_percentage', 'target_tc','label']

    test_df = spark.createDataFrame(data, schema=columns)
    test_df = apply_minmax(test_df, features)

    for j in range(len(features)):
        features[j] = features[j]+'_scaled'
    test_X, label, length, train_id = create_sequences_and_labels(test_df, features, 6)

    print(test_X)
    test_X = torch.tensor(test_X, dtype=torch.float32).cpu()
    len_X = torch.tensor(length, dtype=torch.float32).cpu()
    with torch.no_grad():
        outputs = loaded_model(test_X, len_X).squeeze()
        print(outputs)
    outputs = torch.sigmoid(outputs)
    return outputs

create_test_df()

# COMMAND ----------

year = 2024
TC = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/TC_{year}.parquet')
mgm_reward = spark.read.parquet('dbfs:/mnt/proddatalake/prod/CFA/CFA_overall_lvl')
TC = TC.join(mgm_reward, F.col('playerid') == F.col('MLifeID'), how='inner').select(*TC.columns, 'guest_id')
TC.where('guest_id == 93029205').withColumn('cuml_tc', F.sum(F.col('tiercredit')).over(Window.partitionBy('guest_id').orderBy('transactiondate'))).display()

# COMMAND ----------

Tier = spark.read.parquet(f'dbfs:/mnt/proddatalake/dev/RCX/Tier_History.parquet')
Tier.where('playerid == 82282404').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Overall AUC**

# COMMAND ----------

# loaded_model = torch.load('/Workspace/Users/609399@mgmresorts.com/Tier_Imminent/artifacts/model_2024_v1.pth', map_location='cpu')
# loaded_model.eval()

test_X = torch.tensor(X, dtype=torch.float32).cpu()
len_X = torch.tensor(l, dtype=torch.float32).cpu()
with torch.no_grad():
    outputs = loaded_model(test_X, len_X).squeeze()
outputs = torch.sigmoid(outputs)

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(labels_ls, new_outputs_ls)
auc_score = roc_auc_score(labels_ls, new_outputs_ls)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.grid()
for i in range(0, len(thresholds), len(thresholds)//20):
    plt.text(fpr[i], tpr[i], f'{thresholds[i]:.2f}', fontsize=8, color='red')
#plt.savefig("/Workspace/Users/609399@mgmresorts.com/Tier Imminent/model_performance_metric/sim_2024_roc.png")
plt.show()
plt.close() 

# COMMAND ----------

for i in range(0,len(thresholds)):
    if thresholds[i] < 0.5:
        print(i)
        break

# COMMAND ----------

fpr[8023]

# COMMAND ----------

sum(pos_ls.values())/365

# COMMAND ----------

# MAGIC %md
# MAGIC ## .15 Thresh
# MAGIC - Total Success: 3252
# MAGIC - Total Cover: 14993
# MAGIC - Daily Email: 268
# MAGIC
# MAGIC ## .35 Thresh
# MAGIC - Total Success: 3252
# MAGIC - Total Cover: 24499
# MAGIC - Daily Email: 1335

# COMMAND ----------


