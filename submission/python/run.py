# -*- coding: utf-8 -*-
#!/bin/python3.5

"""
The run.py file produces our final submission into a "submission.csv"
in the data folder
"""

from als import ALS
from sgd import SGD
from helpers import create_csv_submission, load_data

if __name__ == '__main__':
  # Initializing dataset
  print("Loading dataset")
  path_dataset = "data/data_train.csv"
  ratings = load_data(path_dataset)

  # Creating the sub_file with the best prediction
  # prediction, test_rmse = ALS(ratings, None, 3, 0.2, 0.9)
  prediction, test_rmse = SGD(ratings, None, 0.04, 9, 0.1, 0.016)
  create_csv_submission(prediction)
  print("Submission created at data/submission.csv")
