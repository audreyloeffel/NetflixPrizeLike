#!/bin/python3.5
import csv
import re
import numpy as np
from  helpers import load_data, split_data
from lightfm.datasets import fetch_movielens
from plots import plot_raw_data
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

file_path = '../data/data_train.csv'
ratings = load_data(file_path)

num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()

valid_ratings, train, test = split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)

### TRAIN AND TEST ###
# model = LightFM(learning_rate=0.05, loss='bpr')
# model.fit(train, epochs=20)

# train_precision = precision_at_k(model, train, k=10).mean()
# test_precision = precision_at_k(model, test, k=10).mean()

# train_auc = auc_score(model, train).mean()
# test_auc = auc_score(model, test).mean()

### SUBMIT ###
model = LightFM(learning_rate=0.05, loss='bpr')
model.fit(ratings, epochs=20)

indices = []
with open("../data/sampleSubmission.csv", 'r') as sample:
  samples = sample.read().splitlines()[1:]

indices = [ re.match(r'r(\d+?)_c(\d+?),.*?', line, re.DOTALL).groups() for line in samples ]

users = []
items = []
for user, item in indices:
  users.append(user)
  items.append(item)

print(type(users), type(items))
print(len(users), len(items))
users = np.array(users, dtype=np.int32).reshape(-1, 1)
items = np.array(items, dtype=np.int32).reshape(-1, 1)

temp = model.predict(users, items)
# with open("../../data/pred_lightfm.csv", 'w') as csvfile:
#   fieldnames = ['Id', 'Prediction']
#   writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
#   writer.writeheader()
#   for item, user in indices:
#     pred = algo.predict(user, item)
#     writer.writerow({'Id':"r" + item + "_c" + user,'Prediction':pred.est})
