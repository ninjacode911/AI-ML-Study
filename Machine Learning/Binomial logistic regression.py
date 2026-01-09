from sklearn.datasets import load_breast_cancer  # dataset loader (binary classification)
from sklearn.linear_model import LogisticRegression  # logistic regression model
from sklearn.model_selection import train_test_split  # utility to split data into train/test
from sklearn.metrics import accuracy_score  # metric to evaluate classification accuracy

# Load the breast cancer dataset as feature matrix X and target vector y
# This dataset is commonly used for binary classification (malignant vs benign).
X, y = load_breast_cancer(return_X_y=True)

# Split data into training and testing sets.
# test_size=0.20 -> 20% data for testing; random_state=23 ensures reproducible splits.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

# Instantiate the Logistic Regression classifier.
# max_iter=10000 increases the maximum number of iterations to ensure convergence.
# random_state=0 makes the solver's random behavior reproducible.
clf = LogisticRegression(max_iter=10000, random_state=0)

# Train the model on the training data (fit the model parameters).
clf.fit(X_train, y_train)

# Predict on the test set and compute accuracy.
# accuracy_score returns a value between 0 and 1, multiply by 100 to get percentage.
acc = accuracy_score(y_test, clf.predict(X_test)) * 100

# Print a concise summary of model performance (accuracy percentage with 2 decimals).
print(f"Logistic Regression model accuracy: {acc:.2f}%")
