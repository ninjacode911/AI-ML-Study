'''Import the train_test_split function from the sklearn.model_selection module.
 This function is used to split the dataset into training and testing sets.'''
from sklearn.model_selection import train_test_split
'''Import the datasets, linear_model, and metrics modules from the sklearn library.
datasets: Used to load sample datasets.
linear_model: Contains the Logistic Regression model.
metrics: Used for evaluating the model's performance.'''
from sklearn import datasets, linear_model,metrics

'''Load the digits dataset from sklearn. This dataset contains images of handwritten digits (0-9).'''
digits = datasets.load_digits()

'''Extract the feature data (pixel values for each image) from the dataset and assign it to the variable 'x'.'''
x= digits.data
'''Extract the target data (the actual digit each image represents) from the dataset and assign it to the variable 'y'.'''
y= digits.target

'''Split the data into training and testing sets.
x: The feature data.
y: The target data.
test_size=0.4: 40% of the data will be used for testing, and 60% for training.
random_state=1: Ensures that the data is split in the same way every time the code is run, for reproducibility.'''
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=1) 

'''Create an instance of the Logistic Regression model.
max_iter=1000: The maximum number of iterations for the solver to converge.
random_state=0: Ensures that the model initialization is the same every time the code is run, for reproducibility.'''
reg = linear_model.LogisticRegression(max_iter=1000, random_state=0)
'''Train the Logistic Regression model using the training data (X_train and y_train).
The .fit() method learns the relationship between the features and the target variable.'''
reg.fit(X_train, y_train)

'''Use the trained model to make predictions on the test data (X_test).'''
y_pred = reg.predict(X_test)

'''Calculate and print the accuracy of the model.
metrics.accuracy_score(y_test, y_pred): Compares the predicted values (y_pred) with the actual values (y_test) to determine the proportion of correct predictions.
* 100: Converts the accuracy score to a percentage.
:.2f: Formats the percentage to two decimal places.'''
print(f"Logistic Regression Model Accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")
