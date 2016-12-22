# -*- coding: utf-8 -*-
#!/bin/python2.7

"""
The findBestSparkModel.py file is our implementation of ALS using the SPARK
MlLib library, do not forget to install spark before running this file,
and also to change the path below.
"""

from helpers import load_data
import sys
sys.path.append('/usr/local/lib/spark-2.0.2-bin-hadoop2.7/python/')
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import numpy as np
import re
import csv

sCtxt = SparkContext()

# Convert the sparse matrix into RDD containing tuples (user_i, item_i, rating_i)
def fromArrayToRDD(matrix):
  indices = matrix.nonzero()
  acc = []

  for i, j in zip(indices[0], indices[1]):
    acc.append((i, j, matrix[i, j]))

  rdd = sCtxt.parallelize(acc)
  print("######## RDD created")
  return rdd

"""
Iterates over all the possibles parameters and finds the best ones.
"""
if __name__ == '__main__':

  # Initializing dataset
  path_dataset = "data/data_train.csv"
  ratings = load_data(path_dataset) # List((9123, 762) 5.0, ...)
  ratingsRDD = fromArrayToRDD(ratings)

  # Split the dataset into traing and test sets
  training_RDD, validation_RDD, test_RDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)
  validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
  test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

  seed = 5L
  iterations = 10
  regularization_parameter =[i * 0.01 for i in range(9, 20)]
  ranks = [5, 7, 9]
  # errors = [0] * len(regularization_parameter) * len(ranks)
  errors = [[0]*len(regularization_parameter)] * len(ranks)
  ra = 0
  lam = 0

  min_error = float('inf')
  best_lambda = -1
  best_lambda_index = -1
  best_model = None
  best_rank = -1
  best_rank_index = -1

  # Loop over all possible value fr lambda and rank to find the best parameters for our model that minimize the rmse
  for rank in ranks:
    for regParam in regularization_parameter:
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                          lambda_=regParam)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = np.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors[ra][lam] = error
        print 'For lambda %s and rank %s the RMSE is %s' % (regParam, rank, error)
        if error < min_error:
            min_error = error
            best_lambda = regParam
            best_model = model
            best_rank = rank
            best_rank_index = ra
            best_lambda_index = lam
        lam += 1
        with open('python/logs/sparkLogging', 'a') as f:
          f.write("RMSE on testing set: {}, with rank: {}, lambda: {}\n".format(error, rank, regParam))

    lam = 0
    ra += 1

  print 'The best model was trained with lambda %s, rank %s and RMSE: %s' % (best_lambda, best_rank, min_error)

  with open('python/logs/sparkLoggingBest', 'a') as f:
    f.write("RMSE on testing set: {}, with rank: {} at index {}, lambda: {} at index {}\n".format(errors[best_rank_index][best_lambda_index], best_rank, best_lambda_index,  best_lambda, best_lambda_index))

