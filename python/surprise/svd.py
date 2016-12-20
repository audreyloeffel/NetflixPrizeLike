#!/bin/python3.5
from surprise import SVD
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate
from surprise import Reader


file_path = '../../data/data_set.data'
reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
bsl_options = {'method': 'sgd',
               'reg': 9,
               'learning_rate': 0.005,
                }
sim_options = {'name': 'msd',
               'shrinkage': 0,
               'user_based': 'True',
               'min_support': 10,
                }
# algo = KNNWithMeans(bsl_options=bsl_options, sim_options=sim_options)
algo = SVD(n_factors=10, n_epochs=20, lr_all=0.0063, reg_pu=0.04, reg_qi=0.08)
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print(perf)
