'''Import the load_breast_cancer function from the sklearn.datasets module. This function is used to load the breast cancer dataset, which is a classic binary classification dataset.'''
from sklearn.datasets import load_breast_cancer
'''Import the LogisticRegression class from the sklearn.linear_model module. This class is used to create a logistic regression model.'''
from sklearn.linear_model import LogisticRegression
'''Import the train_test_split function from the sklearn.model_selection module. This function is used to split the dataset into training and testing sets.'''
from sklearn.model_selection import train_test_split
'''Import the accuracy_score function from the sklearn.metrics module. This function is used to calculate the accuracy of the classification model.'''
from sklearn.metrics import accuracy_score

'''Load the breast cancer dataset.
return_X_y=True ensures that the function returns the feature data (X) and the target data (y) separately.
X will contain the features (e.g., tumor radius, texture), and y will contain the corresponding labels (0 for malignant, 1 for benign).'''
X, y = load_breast_cancer(return_X_y=True)

'''Split the dataset into training and testing sets.
X: The feature data.
y: The target data.
test_size=0.20: 20% of the data will be used for testing, and 80% for training.
random_state=23: Ensures that the data is split in the same way every time the code is run, for reproducibility.'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

'''Create an instance of the Logistic Regression model.
This initializes the classifier.
max_iter=10000: The maximum number of iterations for the solver to converge. This is set to a high value to ensure the model has enough iterations to find the best solution.
random_state=0: Ensures that the model initialization is the same every time the code is run, for reproducibility.'''
clf = LogisticRegression(max_iter=10000, random_state=0)

'''Train the Logistic Regression model using the training data (X_train and y_train).
The .fit() method learns the relationship between the features and the target variable from the training data.'''
clf.fit(X_train, y_train)

'''Use the trained model to make predictions on the test data (X_test).
The .predict() method returns the predicted labels for the input data.
Then, calculate the accuracy of the model by comparing the predicted labels with the true labels (y_test).
accuracy_score returns the proportion of correct predictions.
Multiply by 100 to express the accuracy as a percentage.'''
acc = accuracy_score(y_test, clf.predict(X_test)) * 100

'''Print the accuracy of the Logistic Regression model.
The f-string formats the output to display the accuracy as a percentage with two decimal places.'''
print(f"Logistic Regression model accuracy: {acc:.2f}%")