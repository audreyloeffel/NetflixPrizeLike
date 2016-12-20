import numpy as np
from helpers import compute_error
from ml_helpers import init_MF


def SGD(train, test, gamma, k, lambda_u, lambda_i):
    """matrix factorization by SGD."""
    # define parameters
    gamma_init = gamma
    num_features = k # K in the lecture notes
    lambda_user = lambda_u
    lambda_item = lambda_i
    num_epochs = 30     # number of full passes through the train set
    errors = [0]

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2

        counter = 0
        prediction = item_features.dot(user_features.T)
        for d, n in nz_train:

            g_w = (train[d, n] - item_features[d].dot(user_features[n].T))*(user_features[n,:])
            g_w = - (g_w - (lambda_item * item_features[d,:]))
            g_z = (train[d, n].T - item_features[d].dot(user_features[n].T).T)*(item_features[d,:])
            g_z = - (g_z - (lambda_user * user_features[n,:]))

            g_w = np.squeeze(np.asarray(g_w))
            g_z = np.squeeze(np.asarray(g_z))

            item_features[d,:] -= gamma*g_w
            user_features[n,:] -= gamma*g_z


        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))

        errors.append(rmse)

    rmse = compute_error(test, user_features, item_features, nz_test)
    with open('overnight_logging_sgd', 'a') as f:
      f.write("RMSE on testing set: {}, with g: {}, k: {}, l_u: {}, l_i: {}\n".format(rmse_test, gamma, num_features, lambda_user, lambda_item))
    # print("RMSE on test data: {} with gamma={}, k={}, lambda_u={}, lambda_i={}.".format(rmse, gamma_init, k, lambda_u, lambda_i))
    return item_features.dot(user_features.T), rmse_test
