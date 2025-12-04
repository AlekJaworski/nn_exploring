import numpy as np
import utils as utils

# generating a visualisable 2d spiral data set
X, y = utils.generate_data()

X_train = X[:300]
X_train = [np.reshape(x, (2, 1)) for x in X_train]
X_test = X[200:]
X_test = [np.reshape(x, (2, 1)) for x in X_test]

# in the future implementation will split the test into the train and test data.
Y_train = y[:300]
Y_train = [np.reshape(utils.num_to_list(z), (3, 1)) for z in Y_train]
Y_test = y[200:]
Y_test = [np.reshape(utils.num_to_list(z), (3, 1)) for z in Y_test]

# preparing the data
train_data = list(zip(X_train, Y_train))
test_data = list(zip(X_test, Y_test))
# utils.visualise(X, y)  # Comment out visualization for testing

# training the example net.
example_net = utils.initialize_new()
utils.train_ex(example_net, train_data, 300, 1, 10)