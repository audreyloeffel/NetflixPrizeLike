from als import ALS
from helpers import create_csv_submission, load_data

path_dataset = "../data/data_train.csv"
ratings = load_data(path_dataset)

valid_ratings, train, test = split_data(
    ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)

prediction = ALS(ratings, test, 3, 0.2, 0.9)

create_csv_submission(prediction)
