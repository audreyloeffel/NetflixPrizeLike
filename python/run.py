#!/bin/python3.5

from als import ALS
from helpers import create_csv_submission, load_data, split_data
import numpy as np
import multiprocessing
import logging


if __name__ == '__main__':
  path_dataset = "../data/data_train.csv"
  ratings = load_data(path_dataset)

  num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
  num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()

  valid_ratings, train, test = split_data(
      ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)

  best_parameters = [np.inf, 0, 0, 0]
  pool = multiprocessing.Pool(processes=3)
  args_list = []
  for k in range(3, 11):
    for lambda_user in [0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]:
      for lambda_item in [0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]:
        args_list.append((train, test, k, lambda_user, lambda_item))

  pool.starmap(ALS, args_list)
        # prediction, test_rmse = ALS(train, test, k, lambda_user, lambda_item)
        # if test_rmse < best_parameters[0]:
        #   print("New best parameters are with RMSE on the test set of:", test_rmse, "are: k =", k, "lambda_user:", lambda_user, "lambda_item:", lambda_item)
        #   best_parameters = [test_rmse, k, lambda_user, lambda_item]

# print("Final best parameters are:", best_parameters)


# prediction, test_rmse = ALS(ratings, test, 3, 0.2, 0.9)

# create_csv_submission(prediction)
