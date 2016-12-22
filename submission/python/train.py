# -*- coding: utf-8 -*-
#!/bin/python3.5

"""
The train.py file trains on a dataset using SGD or ALS asynchronously
in order to be able to run multiple instances at a time to test
several parameter setups.
"""

from als import run_als_asynchronously
from sgd import run_sgd_asynchronously
from helpers import load_data, split_data
import numpy as np

if __name__ == '__main__':
  # Initializing dataset
  path_dataset = "data/data_train.csv"
  ratings = load_data(path_dataset)

  num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
  num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()

  valid_ratings, train, test = split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)

  # Uncomment these lines if you want to train on ALS and input your parameter tuples
  # args_list = [(train, test, 9, 0.1, 0.014), (train, test, 9, 0.1, 0.016), (train, test, 9, 0.105, 0.01)]
  # run_als_asynchronously(args_list)

  # Uncomment these lines if you want to train on SGD and input your parameter tuples
  args_list = [(train, test, 0.04, 9, 0.1, 0.014), (train, test, 0.04, 9, 0.1, 0.016), (train, test, 0.04, 9, 0.105, 0.01)]
  run_sgd_asynchronously(args_list)
