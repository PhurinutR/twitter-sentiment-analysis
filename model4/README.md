# **Model 4 Implementation Guide**

Author: So Ching Hang

Core Architecture: CountVectorizer embedding model with Decision Tree classifier model


## **Prerequisites**

1. Clone this GitHub repository into your local machine.
2. Install required Python libraries:

```bash
pip install scikit-learn pandas joblib
```

3. Ensure you run Python commands **from the repository root**, so the package is importable.

---

# **Using the CountVectorizer + Decision Tree Package for Training**

The main training function is inside:

```
model4/package/decision_tree_model.py
```

The class you will use is:

```python
DecisionTreeSentiment
```

A working demo script is provided:

```
python -m model4.package.test_train
```

### **Sample Usage**

```python
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
```

This function performs:

* CountVectorizer construction
* Decision Tree training
* Optional GridSearchCV hyperparameter tuning
* Reporting CV accuracy

Nothing is returned except the best parameters and cross-validation score.

---

#  **Saving and Loading the Decision Tree Model**

Inside `decision_tree_model.py`, two functions support model persistence:

* The package saves:

  * `vectorizer.joblib`
  * `scikit_tree_model.joblib`

Run the test example:

```
python -m model4.package.test_load
```

---

## **Sample Usage: Load Saved Model**

```python
from model4.package import load_saved_model

data = load_saved_model(
    saved_dir='model4/saved_model'
)

print(data)
```

You will receive a dictionary:

```python
{
    "model": <DecisionTreeClassifier object>,
    "embedding": <CountVectorizer object>
}
```

### Using the loaded embedding + model

```python
vectorizer = data['embedding']
model = data['model']

sample = vectorizer.transform(["I love this product!", "Terrible service."])

predictions = model.predict(sample)
print(predictions)
```

A full example is provided in `test_load.py`.

---

# **Evaluating Saved Models**

The function `evaluate_saved_model()` lets you compute accuracy and loss on both training and test datasets using the **loaded** model.

Example:

```python
eval = evaluate_saved_model(
    models=data,
    train_csv='Twitter_data/traindata.csv',
    test_csv='Twitter_data/testdata.csv'
)

print(eval)
```

You may access metrics through:

```python
eval['train']['accuracy']
eval['train']['loss']
eval['test']['accuracy']
eval['test']['loss']
```

A demonstration is also in:

```
python -m model4.package.test_load
```
