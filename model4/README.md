# **Model 4 Implementation Guide**

Author: *So Ching Hang*

Core Architecture: CountVectorizer embedding model with Decision Tree classifier model

## **Prerequisites**

1. Clone this GitHub repository into your local machine.
2. Install required Python libraries:

```bash
pip install scikit-learn pandas joblib
```

3. Run all Python commands from the repository root directory, so the package imports correctly.

---

# **Using the CountVectorizer + Decision Tree Package for Training**

The main training class is located in:

```
model4/package/decision_tree_model.py
```

The class you will use:

```python
DecisionTreeSentiment
```

A working demo script is provided:

```
python -m model4.package.test_train
```

### **Sample Usage**

```python
from model4.package import DecisionTreeSentiment, load_dataset, clean_text

# Load CSV data
X_train, y_train, X_test, y_test = load_dataset(
    'Twitter_data/traindata.csv',
    'Twitter_data/testdata.csv'
)

# Clean text
X_train = X_train.apply(clean_text)
X_test = X_test.apply(clean_text)

# Create model
model = DecisionTreeSentiment()

# Train model with hyperparameter tuning (GridSearchCV)
best_params, best_cv_score = model.train(X_train, y_train)

# Evaluate final model
accuracy, error = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
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
from model4.package.decision_tree_model import load_saved_model

data = load_saved_model(saved_dir='model4/saved_model')

```

You will receive:

```python
{
    "model": <Pipeline object>,         # vectorizer + classifier
    "embedding": <CountVectorizer>      # extracted from pipeline
}
```

Example: Predict using the loaded model

```python
vectorizer = data['embedding']
model = data['model']

cleaned = ["i love this product", "terrible service"]
pred = model.predict(cleaned)
print(pred)

```


# **Evaluating Saved Models**

If you implemented evaluate_saved_model() (not shown in your package but mentioned):

```python
eval = evaluate_saved_model(
    models=data,
    train_csv='Twitter_data/traindata.csv',
    test_csv='Twitter_data/testdata.csv'
)

print(eval)
```

Access metrics:

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
