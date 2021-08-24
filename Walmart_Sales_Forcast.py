# Databricks notebook source
# configs={"fs.azure.account.auth.type": "OAuth",
#         "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
#         "fs.azure.account.oauth2.client.id": "*****",
#         "fs.azure.account.oauth2.client.secret": "******",
#         "fs.azure.account.oauth2.client.endpoint": "*******"}

# # Optionally, you can add <directory-name> to the source URI of your mount point.
# dbutils.fs.mount(
#   source = "abfss://data@******.dfs.core.windows.net/",
#   mount_point = "/mnt/datalakegen2",
#   extra_configs = configs)

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
import numpy as np
 


# COMMAND ----------

#import data
df_train= spark.read.csv("/mnt/datalakegen2/train.csv",header=True,inferSchema=True)
df_features=spark.read.csv("/mnt/datalakegen2/features.csv",header=True,inferSchema=True)
df_stores=spark.read.csv("/mnt/datalakegen2/stores.csv",header=True,inferSchema=True)
df_test=spark.read.csv("/mnt/datalakegen2/test.csv",header=True,inferSchema=True)
#merge data
df_temp=df_features.join(df_stores,['Store'],how='inner')
df_train_full=df_train.join(df_temp,['Store','Date','IsHoliday'],how='inner')
df_train_full.show()

# COMMAND ----------

#convert Type from string to integer
rdd2=df_train_full.rdd.map(lambda x: (x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13], 3 if x[14] == 'A' else(2 if x[14] == 'B' else 1),x[15]))
df=rdd2.toDF(df_train_full.columns)

# COMMAND ----------

#assert correct Schema
df.createOrReplaceTempView("TypeChange")
df=sqlContext.sql("select int(Store) ,TIMESTAMP(Date),BOOLEAN(IsHoliday),int(Dept),float(Weekly_Sales),float(Temperature),float(Fuel_Price),float(MarkDown1),float(MarkDown2),float(MarkDown3),float(MarkDown4),float(MarkDown5),float(CPI),float(Unemployment),int(Type),float(size) from TypeChange")
# add week and year columns 
df=df.withColumn("week",weekofyear(df.Date))
df=df.withColumn("year",year(df.Date))
df.show()

# COMMAND ----------

df.write.mode("overwrite").parquet("/tmp/df_new.parquet")

# COMMAND ----------

df=spark.read.parquet("/tmp/df_new.parquet")

# COMMAND ----------

#null values 
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()


# COMMAND ----------

import seaborn as sns 
import matplotlib.pyplot as plt

sns.set(style='white')
corr = df.toPandas().corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
'''
numpy. triu (m, k=0)[source] Upper triangle of an array. 
Return a copy of an array with the elements below the k-th diagonal zeroed.
'''
f, ax = plt.subplots(figsize=(20,15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.title("Correlation Matrix", fontsize=20)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink":1}, annot=True)

plt.show()

# COMMAND ----------

#markdown 1-5 will be removed given the high null count
#Fuel_price will be removed due to the high correlation with year
# Multi-collinearity is not a big problem for non-parmetric regression


# COMMAND ----------

# to insure that ordinary squared error loss is the best way to estimate a regression tree for our model let's check the assumtions of heteroscedasticity and normality
#normality 

#heteroscdasity

# COMMAND ----------

df_random=df.select(['Weekly_Sales','Store','Dept','IsHoliday','Size','Week','Type','Year','Temperature','CPI','Unemployment'])

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

#assemble the training vector 
(trainingData, testData) = df_random.randomSplit([0.8, 0.2])
assembler = VectorAssembler(inputCols=df_random.columns, outputCol="features")
trainingData_Assembler=assembler.transform(trainingData)
testData_Assembler=assembler.transform(testData)

# COMMAND ----------

#define the function to minimize 

def hyper_min(params):
    params = {'maxDepth': int(params['maxDepth']), 
                  'numTrees': int(params['numTrees']), 
                 'featureSubsetStrategy': str(int(params['featureSubsetStrategy']))}
    rf = RandomForestRegressor(labelCol="Weekly_Sales", featuresCol="features",**params)
    rf_model=rf.fit(trainingData_Assembler)
    prediction=rf_model.transform(testData_Assembler)
    score=RegressionEvaluator(labelCol="Weekly_Sales", predictionCol="prediction", metricName="rmse").evaluate(prediction)
    return score


# COMMAND ----------

#define search space 
n_iter=10
features=len(df_random.columns)-1
space={'maxDepth': hp.quniform('maxDepth', 5, 20, 1),
       'numTrees' : hp.quniform('numTrees', 2, 20, 1),
       'featureSubsetStrategy': hp.quniform('featureSubsetStrategy', 1, features, 1)
      }
#select search algorithm
import mlflow
with mlflow.start_run():

#since we are using the spark.ml library we do not need spark trials 
  best=fmin(fn=hyper_min,space=space,algo=tpe.suggest,max_evals=n_iter)

# COMMAND ----------

#final model
rf=RandomForestRegressor(labelCol="Weekly_Sales", featuresCol="features", maxDepth=int(best['maxDepth']),
                      numTrees=int(best['numTrees']))
model=rf.fit(trainingData_Assembler)
prediction=rf_model.transform(testData_Assembler)
score=RegressionEvaluator(labelCol="Weekly_Sales", predictionCol="prediction", metricName="rmse").evaluate(prediction)
print(score)
