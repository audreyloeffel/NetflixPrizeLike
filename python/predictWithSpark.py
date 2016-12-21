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
  return rdd


if __name__ == '__main__':
  
  # Initializing dataset
  path_dataset = "../data/data_train.csv"
  ratings = load_data(path_dataset)
  ratingsRDD = fromArrayToRDD(ratings)

  # Train a model using ALS with the best parameters we found with "findBestSparkModel.py"
  model = ALS.train(ratingsRDD, rank = 3, seed=5L, iterations=10, lambda_=0.08)

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
