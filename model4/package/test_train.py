from model4.package import DecisionTreeSentiment
from model4.package import load_dataset

# Load CSV data
X_train, y_train, X_test, y_test = load_dataset(
    'Twitter_data/traindata.csv',
    'Twitter_data/testdata.csv'
)

# Create model
model = DecisionTreeSentiment()

# Train model using internal hyperparameter tuning
best_params, best_cv_score = model.train(X_train, y_train)

# Evaluate on test set
acc, error = model.evaluate(X_test, y_test)
print("Accuracy:", acc)
print("Error:", error)
