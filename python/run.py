#!/bin/python3.5

from als import ALS
from sgd import SGD
from helpers import create_csv_submission, load_data, split_data
import numpy as np
import multiprocessing
import logging
import re

def run_als_asynchronously(train, test, state_file_path, ks, lambda_users, lambda_items):
  args_list = logging(state_file_path)

  pool = multiprocessing.Pool(processes=3)
  pool.starmap(ALS, args_list)

def run_sgd_asynchronously(train, test, state_file_path, gammas, ks, lambda_users, lambda_items):
  args_list = logging(state_file_path)

  pool = multiprocessing.Pool(processes=7)
  pool.starmap(SGD, args_list)

def logging(state_file_path):
  # Computing the array of parameters to use
  args_list = []
  with open(state_file_path, 'r') as log:
    data = log.read().splitlines()

  if not state_file_path.endswith("_sgd"):
    already_computed = [ re.match(r'.*?k\:\s(.*?),\sl_u\:\s(.*?),\sl_i\s(.*?)$', line, re.DOTALL).groups() for line in data ]
    already_computed = [(int(k), float(l_u), float(l_i)) for k, l_u, l_i in already_computed]
    for k in ks:
      for lambda_user in lambda_users:
        for lambda_item in lambda_items:
          if (k, lambda_user, lambda_item) not in already_computed:
            args_list.append((train, test, k, lambda_user, lambda_item))
    print("Computing the remaining ", len(args_list))
  else:
    already_computed = [ re.match(r'.*?g\:\s(.*?),\sk\:\s(.*?),\sl_u\:\s(.*?),\sl_i\:\s(.*?)$', line, re.DOTALL).groups() for line in data ]
    already_computed = [(float(g), int(k), float(l_u), float(l_i)) for g, k, l_u, l_i in already_computed]
    for g in gammas:
      for k in ks:
        for lambda_user in lambda_users:
          for lambda_item in lambda_items:
            if (g, k, lambda_user, lambda_item) not in already_computed:
              args_list.append((train, test, g, k, lambda_user, lambda_item))
    print("Computing the remaining ", len(args_list))
  print("Already computed:", len(already_computed))
  return args_list


def create_submission_file_best_param(ratings, test, state_file_path):
  with open(state_file_path, 'r') as log:
    data = log.read().splitlines()

  already_computed = [ re.match(r'.*?\:\s(.*?),\s.*?k\:\s(.*?),\sl_u\:\s(.*?),\sl_i\s(.*?)$', line, re.DOTALL).groups() for line in data ]
  already_computed = [(float(pred), int(k), float(l_u), float(l_i)) for pred, k, l_u, l_i in already_computed]
  best_pred, k, lambda_user, lambda_item = sorted(already_computed, key=lambda tup: tup[0])[0]
  print("The best prediction right now is: {}, with k = {}, lambda_user = {} and lambda_item = {}.\nComputing and outputting to csv file...".format(best_pred, k, lambda_user, lambda_item))
  prediction, test_rmse = ALS(ratings, test, k, lambda_item, lambda_user)

  create_csv_submission(prediction)


if __name__ == '__main__':
  # Initializing dataset
  path_dataset = "../data/data_train.csv"
  ratings = load_data(path_dataset)

  num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
  num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()

  valid_ratings, train, test = split_data(
      ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)

  # Computing the best parameters
  gammas = [0.04]
  ks = [3, 5, 7, 9]
  lambda_users = [0.9, 0.5, 0.3, 0.1]
  lambda_items = [0.009, 0.007, 0.005, 0.01]
  # run_als_asynchronously(train, test, 'overnight_logging', ks, lambda_users, lambda_items)
  run_sgd_asynchronously(train, test, 'overnight_logging_sgd', gammas, ks, lambda_users, lambda_items)


  # Creating the sub_file with the best prediction
  # create_submission_file_best_param(ratings, test, 'overnight_logging')
