# Databricks notebook source
# File location and type
file_name = "/FileStore/tables/winequality_white.csv"
data = spark.read.options(header='true', inferschema='true', delimiter=',').csv(file_name)
data.show(50)

# COMMAND ----------

data.describe().toPandas().transpose()

# COMMAND ----------

import pandas as pd
from pandas.plotting import scatter_matrix
numeric_features = [t[0] for t in data.dtypes if t[1] == 'int' or t[1] == 'double']
sampled_data = data.select(numeric_features).sample(False, 0.8).toPandas()
axs = scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
    display(v,h)
    

# COMMAND ----------

# histogram of shares
import matplotlib.pyplot as plt
from math import log
npts = 400
shares = data.select('quality').take(npts)
yy = [float(y[0]) for y in shares]
logy = [log(y) for y in yy]
f, axes = plt.subplots(1,2)
f.tight_layout()
axes[0].hist(yy, bins=20, log=True)
axes[0].set_title('log-Histogram of quality')
axes[1].hist(logy, bins=20, log=False)
axes[1].set_title('Histogram of log(quality)')
display(f)

# COMMAND ----------

# compute covariances between shares and each other variable
import numpy as np
data_taken = data.take(400)
features = data.columns[1:-1]
featureData = np.array([[float(a) for a in row[1:-1]] for row in data_taken])
logLabelData = np.array([np.log(float(row[-1])) for row in data_taken])
labelData = np.array([float(row[-1]) for row in data_taken])
cov = [np.cov(labelData,feat)[0,1] for feat in featureData.T]
cc = [np.corrcoef(labelData,feat)[0,1] for feat in featureData.T]
logcc = [np.corrcoef(logLabelData,feat)[0,1] for feat in featureData.T]
print ('%30s:   %s \t%s \t    %s' % ('feature','corcoef','cc_log','covar'))
print ('%30s:   %s \t%s \t    %s' % ('=======','=======','======','====='))
for pair in sorted(zip(-np.abs(cc),cc,logcc,cov,features)):
  print ('%30s:    %.3f\t%.3f\t   %.3f' % (pair[4],pair[1],pair[2],pair[3]))

# COMMAND ----------

data.show(data.count())

# COMMAND ----------

print("Nombre de lignes : ",data.count())
print("Nombre de colonnes : ",len(data.columns))

# COMMAND ----------

display(data.describe())

# COMMAND ----------

#features = ["volatile acidity", "citric acid", "residual sugar", "chlorides","free sulfur dioxyde","total sulfur dioxyde","density","pH","sulphates","alcohol"]
#lr_data = data.select(data, *features)
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
assembler = VectorAssembler(inputCols=["citric acid","pH","sulphates","alcohol"],outputCol="features")

lr_data = assembler.transform(data)
lr_data.printSchema()

# COMMAND ----------

#from pyspark.ml.feature import StandardScaler
#from pyspark.ml import Pipeline
#from pyspark.sql.functions import *
#from pyspark.ml.regression import LinearRegression
#(training, test) = lr_data.randomSplit([.7, .3])
#vectorAssembler = VectorAssembler(inputCols=features, outputCol="unscaled_features")
#standardScaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
#lr = LinearRegression(maxIter=10, regParam=.01)
#stages = [vectorAssembler, standardScaler, lr]
#pipeline = Pipeline(stages=stages)
#print(pipeline)
#model = pipeline.fit(training)
#prediction = model.transform(test)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ["citric acid","pH","sulphates","alcohol"], outputCol = "features")
tdata = vectorAssembler.transform(data)
tdata = tdata.select(["features", "quality"])
tdata.show(3)

# COMMAND ----------

splits = tdata.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

# COMMAND ----------

# DBTITLE 1,Regression Linéaire
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
lr = LinearRegression(featuresCol = "features", labelCol="quality", maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
prediction = lr_model.transform(test_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

# COMMAND ----------

# DBTITLE 1,Résultats après application de la regression linéaire
prediction.show()

# COMMAND ----------

# DBTITLE 1,Résultats après application de la regression linéaire
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
eval = RegressionEvaluator(labelCol="quality", predictionCol="prediction", metricName="rmse")

# Root Mean Square Error
rmse = eval.evaluate(prediction)
print("RMSE: %.3f" % rmse)

# Mean Square Error
mse = eval.evaluate(prediction, {eval.metricName: "mse"})
print("MSE: %.3f" % mse)

# Mean Absolute Error
mae = eval.evaluate(prediction, {eval.metricName: "mae"})
print("MAE: %.3f" % mae)

# r2 - coefficient of determination
r2 = eval.evaluate(prediction, {eval.metricName: "r2"})
print("r2: %.3f" %r2)

train_df.describe().show()

trainingSummary = lr_model.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()

# COMMAND ----------

CSV_PATH = "/FileStore/tables/winequality_white.csv"
APP_NAME = "Random Forest Example"
SPARK_URL = "local[*]"
RANDOM_SEED = 13579
TRAINING_DATA_RATIO = 0.7
RF_NUM_TREES = 3
RF_MAX_DEPTH = 4
RF_NUM_BINS = 32

# COMMAND ----------

from pyspark import SparkContext
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName(APP_NAME) \
    .master(SPARK_URL) \
    .getOrCreate()

df = spark.read \
    .options(header = "true", inferschema = "true") \
    .csv(CSV_PATH)

print("Total number of rows: %d" % df.count())

# COMMAND ----------

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

transformed_df = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[0:-1])))

splits = [TRAINING_DATA_RATIO, 1.0 - TRAINING_DATA_RATIO]
train_df, test_df = transformed_df.randomSplit(splits, RANDOM_SEED)

print("Number of training set rows: %d" % train_df.count())
print("Number of test set rows: %d" % test_df.count())

# COMMAND ----------

# DBTITLE 1,Random Forest
from pyspark.mllib.tree import RandomForest
from time import *

start_time = time()

model = RandomForest.trainClassifier(train_df, numClasses=15, categoricalFeaturesInfo={}, \
    numTrees=20, featureSubsetStrategy="auto", impurity="gini", \
    maxDepth=20,seed=RANDOM_SEED)
#RF_MAX_DEPTH
#RF_NUM_TREES
end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)

# COMMAND ----------

# DBTITLE 1,Test Accuracy Random Forest
predictions = model.predict(test_df.map(lambda x: x.features))
labels_and_predictions = test_df.map(lambda x: x.label).zip(predictions)
acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(test_df.count())
print("Model accuracy: %.3f%%" % (acc * 100))

# COMMAND ----------

# DBTITLE 1,Evaluation Random Forest
from pyspark.mllib.evaluation import BinaryClassificationMetrics

start_time = time()

metrics = BinaryClassificationMetrics(labels_and_predictions)
print("Area under Precision/Recall (PR) curve: %.f" % (metrics.areaUnderPR * 100))
print("Area under Receiver Operating Characteristic (ROC) curve: %.3f" % (metrics.areaUnderROC * 100))

end_time = time()
elapsed_time = end_time - start_time
print("Time to evaluate model: %.3f seconds" % elapsed_time)
