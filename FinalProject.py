import argparse
import numpy as np

from time import time

from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics


from pyspark import SparkContext
from pyspark.sql import SparkSession

def readData(input_file,spark_context):
  """  
    Read mnist data from csv file that is structured as
    label1-1, px1-1, px1-2, ..., px28-28
    returns a key value pair where the key is the pixel information
    and value is the label   
  """
  return spark_context.textFile(input_file)\
    .map(eval)\
    .map(lambda data: (Vectors.dense(data[1:]),data[0]))
  
def rddToDf(input_rdd, labels, spark_session):
  """
    Convert RDD to data frame
  """
  return spark_session.createDataFrame(input_rdd, labels)

def kMeans(k, seed):
  """
    Initialize parallelized K-Means object
  """
  return KMeans().setK(k).setSeed(seed)

def trainKmeans(kmeans, train_df):
  startTime = time()
  model = kmeans.fit(train_df)
  endTime = time()
  return (model, endTime - startTime)

def predictKmeans(kMeans_model, test_df):
  startTime = time()
  predictions = kMeans_model.transform(test_df)
  endTime = time()
  return (predictions, endTime - startTime)

def nmi(result_rdd, spark_context):
  """
    Compute nmi using result_rdd, where each item is tuple (class_label, cluster_label)
  """
  
  def add(x,y):
    return x+y

  def info(x):
    if(x == 0.0):
      return 0.0
    return x * np.log2(x)

  def divideNum(val):
    return lambda x: 1.0 * x / val

  numItems = result_rdd.count()
  classCountsRdd = result_rdd.map(lambda data: (data[0], 1)).reduceByKey(add)
  classProb = classCountsRdd.mapValues(divideNum(numItems))
  classEntropy = classProb.mapValues(lambda value: -info(value)).values().reduce(add)
  clusterCountsRdd = result_rdd.map(lambda data: (data[1], 1)).reduceByKey(add).persist()
  clusterCountsObject = dict(clusterCountsRdd.collect())
  clusterProb = clusterCountsRdd.mapValues(divideNum(numItems))
  clusterEntropy = clusterProb.mapValues(lambda value: -info(value)).values().reduce(add)
  clusterCountsRdd.unpersist()
  countMatrix = result_rdd.map(lambda data: (data, 1)).reduceByKey(add)
  classCondCount = countMatrix.map(lambda data: (data[0][1], data[1]))
  classCondProb = classCondCount.map(lambda data: (data[0], divideNum(clusterCountsObject[data[0]])(data[1])))
  classCondEntropy = classCondProb.mapValues(info).reduceByKey(add).map(lambda data: (data[0], -1.0 * data[1] * divideNum(numItems)(clusterCountsObject[data[0]]))).values().reduce(add)
  mi = classEntropy - classCondEntropy
  nmi = 2.0 * mi / (classEntropy + clusterEntropy)
  return nmi

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'KMeans clustering on MNIST.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--master', default="local[20]", help="Specify the deploy mode")
  parser.add_argument('--k', type=int, default=10, help='Number of clusters')
  parser.add_argument('--p', type=int, default=10, help='Number of partitions')
  parser.add_argument('--seed', type=int, default=1, help='Initial seed for kmeans')
  args = parser.parse_args()
  sc = SparkContext(args.master, appName='KMeans clustering on MNIST')
  sc.setLogLevel('warn')
  spark = SparkSession(sc)
  train_set = readData('mnist_train.csv', sc)
  train_df = rddToDf(train_set, ['features', 'label'] ,spark).coalesce(args.p)
  test_set = readData('mnist_test.csv', sc)
  test_df = rddToDf(test_set, ['features', 'label'] ,spark).coalesce(args.p)
  kmeans = kMeans(args.k, args.seed)
  model, totalTrainTime = trainKmeans(kmeans, train_df)
  print('training time', totalTrainTime)
  predictions, totalPredictionTime = predictKmeans(model, test_df)
  print('prediction time', totalPredictionTime)
  filteredData = predictions.select('label','prediction').rdd
  nmi_score = nmi(filteredData, sc)
  print('nmi_score', nmi_score)
