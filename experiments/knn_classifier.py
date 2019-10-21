import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from kutu.knn_classifier import KNeighborsClassifierNumpy
from sklearn.metrics import r2_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate random data-set
X, y = make_classification(n_samples=1000, 
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
neighbors_models = [ ('scikit', KNeighborsClassifier(n_neighbors=5)),
                    ('kutu-numpy', KNeighborsClassifierNumpy(n_neighbors=5))]
                    

for model_type, neighbors_model in neighbors_models:

    # Fit the data(train the model)
    neighbors_model.fit(x_train, y_train)

    # Predict
    y_predicted = neighbors_model.predict(x_test)

    # Model evaluation
    acc = accuracy_score(y_test, y_predicted)

    # Print values
    print('Model type: %s' % model_type)
    print('Accuracy: ', acc)

    #print(y_test)
    #print(y_predicted)
    