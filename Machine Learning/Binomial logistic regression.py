'''Imports the 'load_breast_cancer' function from the 'sklearn.datasets' module. This function is used to load the breast cancer dataset, which is included with scikit-learn.'''
from sklearn.datasets import load_breast_cancer

'''Imports the 'LogisticRegression' class from the 'sklearn.linear_model' module. This class is used to create a logistic regression model.'''
from sklearn.linear_model import LogisticRegression
'''Imports the 'train_test_split' function from the 'sklearn.model_selection' module. This function is used to split the dataset into training and testing sets.'''
from sklearn.model_selection import train_test_split

'''Imports the 'accuracy_score' function from the 'sklearn.metrics' module. This function is used to calculate the accuracy of the model's predictions.'''
from sklearn.metrics import accuracy_score

'''Loads the breast cancer dataset. 'return_X_y=True' ensures that the data is returned as two separate variables:
- X: A variable holding the features of the dataset (the characteristics of the tumors).
- y: A variable holding the target labels (whether a tumor is malignant or benign).'''
X, y = load_breast_cancer(return_X_y=True)

'''Splits the dataset into training and testing sets.
- X and y are the features and labels we just loaded.
- 'test_size=0.20' means that 20% of the data will be used for testing, and the remaining 80% will be used for training.
- 'random_state=23' is a seed for the random number generator. Using a specific number ensures that the data is split in the exact same way every time you run this code. This is important for getting reproducible results.'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

'''Creates an instance of the LogisticRegression model.
- 'max_iter=10000' sets the maximum number of iterations for the solver to converge. This can be increased if the model fails to converge.
- 'random_state=0' is used to ensure that the model's internal random processes are the same every time. This helps in getting reproducible results from the model itself.'''
clf = LogisticRegression(max_iter=10000, random_state=0)

'''Trains the logistic regression model. The 'fit' method teaches the model to find the relationship between the features (X_train) and the target labels (y_train).'''
clf.fit(X_train, y_train)

'''Calculates the accuracy of the model.
- 'clf.predict(X_test)' uses the trained model to make predictions on the test data (X_test).
- 'accuracy_score' then compares these predictions to the actual labels (y_test).
- The result is multiplied by 100 to express it as a percentage.'''
acc = accuracy_score(y_test, clf.predict(X_test)) * 100

'''Prints the accuracy of the model to the console.
- The 'f' before the string indicates a formatted string.
- '{acc:.2f}' is a placeholder that will be replaced by the value of the 'acc' variable, formatted to two decimal places.'''
print(f"Logistic Regression model accuracy: {acc:.2f}%")
