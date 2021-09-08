# Databricks notebook source
# configs={"fs.azure.account.auth.type": "OAuth",
#         "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
#         "fs.azure.account.oauth2.client.id": "e52a57e8-947d-4c40-9952-3cf9dd3f83d5",
#         "fs.azure.account.oauth2.client.secret": "66anx3S57049LRk7kHoYDcWi-qyC__-6oA",
#         "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/f729dc92-7f20-4c3a-a702-208d6bb1299c/oauth2/token"}

# # Optionally, you can add <directory-name> to the source URI of your mount point.
# dbutils.fs.mount(
#   source = "abfss://data@wal203.dfs.core.windows.net/",
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
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
import cloudpickle

#assemble the training vector 
(trainingData, testData) = df_random.randomSplit([0.8, 0.2])
assembler = VectorAssembler(inputCols=df_random.columns, outputCol="features")
trainingData_Assembler=assembler.transform(trainingData)
testData_Assembler=assembler.transform(testData)

# COMMAND ----------

#define the function to minimize 
import mlflow
def hyper_min(params):
    with mlflow.start_run(nested=True):
      params = {'maxDepth': int(params['maxDepth']), 
                    'numTrees': int(params['numTrees']), 
                   'featureSubsetStrategy': str(int(params['featureSubsetStrategy']))}
      rf = RandomForestRegressor(labelCol="Weekly_Sales", featuresCol="features",**params)
      rf_model=rf.fit(trainingData_Assembler)
      prediction=rf_model.transform(testData_Assembler)
      score=RegressionEvaluator(labelCol="Weekly_Sales", predictionCol="prediction", metricName="rmse").evaluate(prediction)
      mlflow.log_param('maxDepth', int(params['maxDepth']))
      mlflow.log_param('numTrees', int(params['numTrees']))
      mlflow.log_param('featureSubsetStrategy', str(int(params['featureSubsetStrategy'])))
      mlflow.log_metric('RMSE',score)

    return score

#define search space 
n_iter=15
features=len(df_random.columns)-1
space={'maxDepth': hp.quniform('maxDepth', 5, 20, 1),
       'numTrees' : hp.quniform('numTrees', 2, 20, 1),
       'featureSubsetStrategy': hp.quniform('featureSubsetStrategy', 1, features, 1)
      }
#select search algorithm

with mlflow.start_run(run_name='untuned_random_forest'):

#since we are using the spark.ml library we do not need spark trials 
  best=fmin(fn=hyper_min,space=space,algo=tpe.suggest,max_evals=n_iter)



# COMMAND ----------

best={'featureSubsetStrategy': 7.0, 'maxDepth': 12.0, 'numTrees': 20.0}


# COMMAND ----------

#final baseline model
import mlflow.pyfunc
with mlflow.start_run(run_name='final_random_forest'):

  rf=RandomForestRegressor(labelCol="Weekly_Sales", featuresCol="features", maxDepth=int(best['maxDepth']),
                        numTrees=int(best['numTrees']))
  rf_model=rf.fit(trainingData_Assembler)
  prediction=rf_model.transform(testData_Assembler)
  score=RegressionEvaluator(labelCol="Weekly_Sales", predictionCol="prediction", metricName="rmse").evaluate(prediction)
  mlflow.log_metric('RMSE',score)  
  run_id = mlflow.active_run().info.run_id
  
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)


# COMMAND ----------

#run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "final_random_forest"').iloc[0].run_id
#print(run_id)

# COMMAND ----------

#model registery
model_name="walmart_forcast"
#model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)
model_version = mlflow.register_model("runs:/dd02b9e617b34202839fea402a824a78/random_forest_model", model_name)


# COMMAND ----------

from mlflow.tracking import MlflowClient
 
client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

# new model
from hyperopt.pyll import scope
from math import exp
import mlflow.lightgbm
import lightgbm as lgb
 
#assemble data
Y_train,Y_test =trainingData.select['Weekly_Sales'],testData.select['Weekly_Sales']
(X_train, X_test)=(trainingData.drop['Weekly_Sales'], testData.drop['Weekly_Sales']) 

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'seed': 123, # Set a seed for deterministic training
}

#define the model  
def train_model(params):
  mlflow.lightgbm.autolog()
  with mlflow.start_run(nested=True):
    gbm = lgb.LGBMRegressor(**params)
    booster = gbm.fit(X_train,Y_train, num_boost_round=200,
                        eval_set=[(X_test, Y_test)], early_stopping_rounds=20)
    predictions_test = booster.predict(test)
    mean_squared_error_score = mean_squared_error(y_test, predictions_test)
    mlflow.log_metric('mean_squared_error', mean_squared_error_score)
 
    signature = infer_signature(X_train, booster.predict(train))
    mlflow.lightgbm.log_model(booster, "model", signature=signature)
    
    return {'status': STATUS_OK, 'loss': -1*mean_squared_error_score, 'booster': booster.attributes()}

#light gbm is not spark optimized so we need to use spark trials 
spark_trials = SparkTrials(parallelism=10)
 
with mlflow.start_run(run_name='LGBM_models'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=96,
    trials=spark_trials, 
    rstate=np.random.RandomState(123)
  )


# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.mean_squared_error DESC']).iloc[0]
print(f'RMSE of Best Run: {best_run["metrics.mean_squared_error"]}')


# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)


# COMMAND ----------

#archive old model and move new model to production

client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Archived"
)
 
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage="Production"
)
