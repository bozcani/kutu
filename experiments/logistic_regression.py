import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from kutu.logistic_regression import LogisticRegressionNumpy
from sklearn.metrics import r2_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate random data-set
X, y = make_classification(n_samples=100, 
                            n_features=2, # num. of total features
                            n_informative=2, # num. of informative features
                            n_redundant=0, # num. of redundant features
                            n_repeated=0, # num. of repeated features
                            n_classes=2) # num. of classes

# Plot the dataset
# plt.scatter(X[:,0], X[:,1], c=y)
# plt.show()

# Split train and test sets.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(x_train.shape)
print(y_train.shape)

# Models to compare
linear_models = [ ('scikit', LogisticRegression(solver='lbfgs', max_iter=1000)),
                    ('kutu-numpy', LogisticRegressionNumpy())]

for model_type, linear_model in linear_models:

    # Fit the data(train the model)
    linear_model.fit(x_train, y_train)

    # Predict
    y_predicted = linear_model.predict(x_test)

    # Model evaluation
    acc = accuracy_score(y_test, y_predicted)

    # Print values
    print('Model type: %s' % model_type)
    print('Slope:' ,linear_model.coef_)
    print('Intercept:', linear_model.intercept_)
    print('Accuracy: ', acc)

    #print(y_test)
    #print(y_predicted)
    