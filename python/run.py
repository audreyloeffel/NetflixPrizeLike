from als import ALS
from helpers import create_csv_submission, load_data, split_data
import numpy as np

path_dataset = "../data/data_train.csv"
ratings = load_data(path_dataset)

num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()

valid_ratings, train, test = split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)

prediction = ALS(ratings, test, 3, 0.2, 0.9)

create_csv_submission(prediction)
