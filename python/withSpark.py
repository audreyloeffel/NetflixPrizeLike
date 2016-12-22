# -*- coding: utf-8 -*-
#!/bin/python3.5

from helpers import create_csv_submission, load_data, split_data
import sys
sys.path.append('/usr/local/lib/spark-2.0.2-bin-hadoop2.7/python/')
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import math
import numpy as np

sCtxt = SparkContext()

def fromArrayToRDD(matrix):
  indices = matrix.nonzero()
  acc = []

  for i, j in zip(indices[0], indices[1]):
    acc.append((i, j, matrix[i, j]))

  rdd = sCtxt.parallelize(acc)
  print("######## RDD created")
  return rdd


if __name__ == '__main__':
  # Initializing dataset

  path_dataset = "../data/data_train.csv"
  ratings = load_data(path_dataset) # List((9123, 762) 5.0, ...)
  ratingsRDD = fromArrayToRDD(ratings)

  training_RDD, validation_RDD, test_RDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)
  validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
  test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
  print(validation_for_predict_RDD.take(3))


  seed = 5L
  iterations = 10
  regularization_parameter = 0.01
  ranks = range(2, 12)
  errors = [0] * len(ranks)
  err = 0
  tolerance = 0.02

  min_error = float('inf')
  best_rank = -1
  best_iteration = -1
  for rank in ranks:
      model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                        lambda_=regularization_parameter)
      predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
      rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
      error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
      errors[err] = error
      err += 1
      print 'For rank %s the RMSE is %s' % (rank, error)
      if error < min_error:
          min_error = error
          best_rank = rank

  print 'The best model was trained with rank %s and RMSE: %s' % (best_rank, min_error)

