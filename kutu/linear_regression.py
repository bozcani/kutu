import numpy as np

class LinearRegressionNumpy:
    """Linear Regresion using gradient descent algorithm.
    Implemented in numpy.
    Related equations:
    # hypothesis function: y_pred = xw + b
    # Loss function: J = (1/2m) sum_i^m (h(xi)-yi)^2
    # Take derivative of the loss function w.r.t. the parameters: dw/dJ = (1/m) sum_i^m (h(xi)-yi)x
    """

    def __init__(self):
        self.w_ = None
        self.intercept_ = None
        self.coef_ = None


    def fit(self, x, y, learning_rate=0.5, n_iterations=1000):
        """Fit the training data using Gradient Descent
        Args:
            x (np.array): Training samples, shape = [n_samples, n_features]
            y (np.array): Target samples, shape = [n_samples, n_target_values]
            learning_rate (float, optional): Learning rate. Defaults to 0.5.
            n_iterations (int, optional): Number of iterations i.e., num. of epochs. Defaults to 1000.
        """

        self.cost_ = []

        # Create weight matrix including the bias term
        np.random.seed(0)
        self.w_ = np.random.rand(x.shape[1]+1, y.shape[1])*0.01
        
        # Add 1 to the beginning of input vectors for the bias.
        m = x.shape[0]
        x = np.hstack((np.ones((m,1)), x))

        # Shuffle the training data.
        indices = np.arange(0, m)
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        # Start of training.
        for _ in range(n_iterations):

            # hypothesis function
            y_pred = np.dot(x, self.w_)

            # residuals, i.e. errors
            residuals = y_pred - y

            # Derivatives of the loss function w.r.t weights, dw/dJ:
            gradient_vector = np.dot(x.T, residuals)

            # Update weights
            self.w_ -= (learning_rate / m) * gradient_vector

            # Save loss function. Might be useful.
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)

        # Intercept and slope (coefficients).
        self.intercept_ = self.w_[0]
        self.coef_ = self.w_[1:]


    def predict(self, x):
        """Predicts for test samples.
        Args:
            x (np.array): Test samples, shape = [n_samples, n_features]
        
        Returns:
            np.array: Predicted value, shape = [n_samples, n_target_values]
        """

        # Add 1 vector to the inputs for bias.
        m = x.shape[0]       
        x = np.hstack((np.ones((m,1)), x))

        # Return hypothesis
        return np.dot(x, self.w_)