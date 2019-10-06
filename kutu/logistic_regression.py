import numpy as np

class LogisticRegressionNumpy:
    """Logistic Regresion using gradient descent algorithm.
    Implemented in numpy.
    Related equations:
    # hypothesis function: y_pred = h(x) = sigmoid(xw + b).
    # Loss function: Log loss aka. Binary cross entropy loss
    # J = -1/N sum_i^N (yi*log(h(xi)) + (1-yi)log(1-h(xi)))
    # Take derivative of the loss function w.r.t. the parameters: dJ/dw = 1/N sum_i^N (xi*(h(xi)-yi))
    """

    def __init__(self):
        self.w_ = None
        self.intercept_ = None
        self.coef_ = None


    def fit(self, x, y, learning_rate=0.05, n_iterations=1000):
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
        self.w_ = np.random.rand(x.shape[1]+1, 1)*0.01
        
        # Add 1 to the beginning of input vectors for the bias.
        m = x.shape[0]
        x = np.hstack((np.ones((m,1)), x))

        # Shuffle the training data.
        indices = np.arange(0, m)
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        y = np.reshape(y, (y.shape[0], 1))

        # Start of training.
        for _ in range(n_iterations):

            # hypothesis function
            h = self._sigmoid(np.dot(x, self.w_)) # y = sigmoid(wx+b)

            # Derivatives of the loss function w.r.t weights, dw/dJ:
            gradient_vector = np.dot(x.T, (h - y))

            # Update weights
            self.w_ -= (learning_rate / m) * gradient_vector

            # Save loss function. Might be useful.
            # Cross entropy loss.
            cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
            self.cost_.append(cost)

        # Intercept and slope (coefficients).
        self.intercept_ = self.w_[0]
        self.coef_ = self.w_[1:]


    def _sigmoid(self, x):
        """Sigmoid function."""
        return 1./(1 + np.exp(-x))


    def predict(self, x):
        """Predicts for test samples.
        Args:
            x (np.array): Test samples, shape = [n_samples, n_features]
        
        Returns:
            np.array: Predicted value, shape = [n_samples, ]
        """

        # Add 1 vector to the inputs for bias.
        m = x.shape[0]       
        x = np.hstack((np.ones((m,1)), x))

        # Get predicted values predicted_labels, y = h(x)
        predicted_labels = (self._sigmoid(np.dot(x, self.w_))>0.5) # y = h(x)
        predicted_labels = predicted_labels.astype(np.uint8) # Convert bool to int.
        predicted_labels = predicted_labels.reshape((m,)) # Reshape to 1D array.
        return predicted_labels