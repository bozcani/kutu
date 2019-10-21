import numpy as np

class KNeighborsClassifierNumpy:
    """K Nearest Neighbors for classification.
    Implemented in numpy.
    KNN is a non-parametric classification algorithm (i.e. does not learn anything regarding the data).
    There are pros and cons of lazy learner algorithms:
    Pros:
        -> No assumption about data. Maybe useful for nonlinear data.
        -> Simple to understand and implement
    Cons:
        -> High memory requirement
        -> Slow inference
        -> Sensitive to irrelevant features, noises, and the scale of the data

    """

    def __init__(self, n_neighbors = 5):
        """Default initializer.
        
        Args:
            neighbors (int, optional): Number of nearest neighbors to classify. Defaults to 5.
        """
        self.k = n_neighbors


    def fit(self, x, y):
        """Fit the training data using euclidian distance.
        It actually does nothing since knn is lazy learner.
        Args:
            x (np.array): Training samples, shape = [n_samples, n_features]
            y (np.array): Target samples, shape = [n_samples, n_target_values]
        """

        self.x = x
        self.y = y

    def predict(self, x):
        """Predicts for test samples.
        Args:
            x (np.array): Test samples, shape = [n_samples, n_features]
        
        Returns:
            np.array: Predicted value, shape = [n_samples, ]
        """

        # Num of the test and train samples.
        num_test = x.shape[0]
        num_train = self.x.shape[0]

        # Create a matrix to hold distances.
        dists = np.zeros((num_test, num_train))

        # Calculate distances.
        for i in range(num_test):
            dists[i,:] = np.sqrt( np.sum( ( x[i] - self.x ) ** 2, axis=1) )

        # Predict.
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.y[np.argsort(dists[i])][0:self.k]
            y_pred[i] = np.bincount(closest_y).argmax()
        return y_pred