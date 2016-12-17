#!/bin/python3.5
from surprise import SVD
from surprise import Dataset
from surprise import evaluate
from surprise import Reader


file_path = '../../data/data_set.data'
reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print(perf)
