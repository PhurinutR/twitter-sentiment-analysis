# Model 4 Implementation Guide  
Author: So Ching Hang

Core Architecture: **CountVectorizer text embedding model with Decision Tree classifier**

---

## Prerequisite

1. Clone this GitHub repository into the local PC.  
2. Run the commands provided in the "Getting Started" guide on the home page to install the necessary Python libraries.  
3. Ensure the folder structure of the package remains unchanged so that relative imports work properly.

---

## Using the CountVectorizer + Decision Tree Package for Training

A function to train **CountVectorizer + Decision Tree** and return evaluation metrics is provided by  
`train_model()`, located inside `decision_tree_model.py` under the package folder.

To see how it works, please run `test_train.py` from the repository root directory:

```bash
python -m textmodels.test_train
```

### Sample Usage

```python
from textmodels import DecisionTreeSentiment
from textmodels import load_dataset

# Load dataset
X_train, y_train, X_test, y_test = load_dataset(
    'data/traindata.csv',
    'data/testdata.csv'
)

# Initialize model
model = DecisionTreeSentiment()

# Train model with optional hyperparameters
model.train(
    X_train, 
    y_train,
    params={
        "vec__ngram_range": [(1,1), (1,2)],
        "vec__min_df": [2],
        "clf__max_depth": [10, 15, 20],
        "clf__criterion": ["entropy"]
    }
)

# Evaluate the model
acc, error = model.evaluate(X_test, y_test)
```

This function will return:

- Best hyperparameters found  
- Best cross-validation accuracy  

And the evaluate function will return:

- Test accuracy  
- Test error  

---

## Notes

- The package must be executed from the **root directory** so the `textmodels` package can be imported correctly.
- Training may display verbose logs from scikit-learn. These are normal and should not be treated as errors.

---

## Using the CountVectorizer + Decision Tree Package for Loading Saved Models

Another function in `decision_tree_model.py` is `load_saved_model()`.  
As the name suggests, it loads previously saved embedding and classifier models into memory for inference.

If the `save_model_path` argument in `train()` is set to a valid folder, the package will save:

- `vectorizer.joblib`  
- `scikit_model.joblib`

To see the working demo, run:

```bash
python -m textmodels.test_load
```

### Sample Usage

```python
from textmodels import load_saved_model

data = load_saved_model(
    saved_dir='textmodels/saved_models'  # Required
)

print(data)
```

### Notes

You may access the loaded models via:

```python
vectorizer = data['embedding']
model = data['model']
```

Both objects are **fully usable scikit-learn model objects**.

To perform inference:

```python
sample = vectorizer.transform(["this product is good"])
prediction = model.predict(sample)
```

A complete example is provided in `test_load.py`.

---

## Using the CountVectorizer + Decision Tree Package for Evaluation on Saved Models

The function `evaluate_saved_model()` allows you to directly evaluate loaded models on training and testing datasets.

Run:

```bash
python -m textmodels.test_load
```

### Sample Usage

```python
eval = evaluate_saved_model(
    models=data,                      # dictionary from load_saved_model()
    train_csv='data/traindata.csv',   # Optional
    test_csv='data/testdata.csv'      # Optional
)

print(eval)
```

### Notes

You can access evaluation metrics via:

```python
eval['train']['accuracy']
eval['train']['loss']
eval['test']['accuracy']
eval['test']['loss']
```

Examples of accuracy/loss extraction are provided in `test_load.py`.

## End of README
