# -*- coding: utf-8 -*-
#!/bin/python3.5

from helpers import create_csv_submission, load_data, split_data
import sys
sys.path.append('/usr/local/lib/spark-2.0.2-bin-hadoop2.7/python/')
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS
import math
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


if __name__ == '__main__':

  # Initializing dataset
  path_dataset = "../data/data_train.csv"
  ratings = load_data(path_dataset) # List((9123, 762) 5.0, ...)
  ratingsRDD = fromArrayToRDD(ratings)

  # Split the dataset into traing and test sets
  training_RDD, validation_RDD, test_RDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)
  validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
  test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

  seed = 5L
  iterations = 10
  regularization_parameter =[i * 0.01 for i in range(1, 10)]
  ranks = range(1, 11)
  errors = [0] * len(regularization_parameter) * len(ranks)
  err = 0

  min_error = float('inf')
  best_lambda = -1
  best_model = None
  best_rank = -1

  # Loop over all possible value fr lambda and rank to find the best parameters for our model that minimize the rmse
  for rank in ranks:
    for regParam in regularization_parameter:
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                          lambda_=regParam)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors[err] = error
        err += 1
        print 'For lambda %s and rank %s the RMSE is %s' % (regParam, rank, error)
        if error < min_error:
            min_error = error
            best_lambda = regParam
            best_model = model
            best_rank = rank

  print 'The best model was trained with lambda %s, rank %s and RMSE: %s' % (best_lambda, best_rank, min_error)

  # Parse the submission sample file to get the indices we need to predict
  indices = []
  with open("../data/sampleSubmission.csv", 'r') as sample:
    data = sample.read().splitlines()[1:]
  indices = [ re.match(r'r(\d+?)_c(\d+?),.*?', line, re.DOTALL).groups() for line in data ]
  indicesRDD = sCtxt.parallelize(indices)

  # Predict and write into the sumbmission file
  with open("../data/submission.csv", 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    predictions = model.predictAll(indicesRDD)
    tuples = predictions.map(lambda p: (p[0], p[1], p[2])).collect()
    for i in range(len(tuples)):
      writer.writerow({'Id':"r" + str(tuples[i][0]+1) + "_c" + str(tuples[i][1]+1),'Prediction':str(tuples[i][2])})
