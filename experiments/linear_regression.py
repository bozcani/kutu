import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from kutu.linear_regression import LinearRegressionNumpy
from sklearn.metrics import r2_score

# Generate random data-set
np.random.seed(0)
x = np.random.rand(100, 1)
y = 7 + 4 * x + np.random.rand(100, 1)

# Models to compare
regression_models = [ ('scikit', LinearRegression()), 
                    ('kutu-numpy',LinearRegressionNumpy())]

for model_type, regression_model in regression_models:

    # Fit the data(train the model)
    regression_model.fit(x, y)

    # Predict
    y_predicted = regression_model.predict(x)

    # Model evaluation
    rmse = np.sum((y - y_predicted)**2)
    r2 = r2_score(y, y_predicted)

    # Print values
    print('Model type: %s' % model_type)
    print('Slope:' ,regression_model.coef_)
    print('Intercept:', regression_model.intercept_)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)

    # Data points
    plt.scatter(x, y, s=10)
    plt.title(model_type)
    plt.xlabel('x')
    plt.ylabel('y')

    # Predicted values
    plt.plot(x, y_predicted, color='r')
    plt.show()
