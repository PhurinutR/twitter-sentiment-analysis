# Model 5 Implementation Guide
Core architecture: **TF-IDF** embedding model with **Naive Bayes** classifier model<br>
Author: *Marcus KWAN TH*<br>

## Prerequisite
1. Clone this GitHub repository into the local PC.
2. Run the commands provided in the "Getting Started" guide in the home page to install necessary libraries.

## Using the TF-IDF with Naive Bayes Package for **Training and Evaluation**

A function to run TF-IDF + Naive Bayes and return the accuracy with loss metrics are provided by `train_model()`, within a package `model.py` under `package` folder.

Please run `test_train.py` from the repository root directory to see how it works:

```bash
python -m model5.package.test_train
```

### Sample Usage:

```python
from model5.package import train_model

res = train_model(
	'Twitter_data/traindata.csv',
	'Twitter_data/testdata.csv',
    tfidf_parameters = {"min_df": 4, "max_df": 0.95, "ngram_range": (1, 2), "use_idf": True} # Optional: Sample paramters for TF-IDF embedding
    nb_parameters = {"alpha": 1.0}  # Optional: Sample parameters for scikit-learn's Naive Bayes classifier
	save_model_path='model5/package/saved_models'  # Optional: persist embedding model and classifier model
)
print(res)
```

### Notes:
- The package is run from the root directory so the `util` package is importable.
- Some verbose and warning from PySpark may appear in the console output. They are normal and no need to be worried about.

## Using the TF-IDF with Naive Bayes Package for **Evaluation on Saved Models**

Another function in `model.py` is `evaluate_saved_model()`. As the name suggest, it loads the saved model from a directory then perform eveluations with accuracy and loss metrics.

Please run `test_load.py` from the repository root directory to see the result:

```bash
python -m model5.package.test_load
```

### Sample Usage:

```python
    eval = evaluate_saved_model(
        saved_dir='model5/saved_model', 			# Required.
        train_csv='Twitter_data/traindata.csv', 	# Optional, no training result will be returned if set to None.
        test_csv='Twitter_data/testdata.csv'		# Optional, no testing result will be returned if set to None.
    )
	print(eval)
```
### Notes:
	- You may access the evaluated metrics using `eval['train']` and `data['test']`. To access the accuracy and loss metric separately, append `['accuracy']` or `['loss']`. An example is provided in the `test_load.py`.

## Using the TF-IDF with Naive Bayes Package for **Loading the Saved Model**

Another function in `model.py` is `load_saved_model()`. As the name suggest, it loads the saved model from a directory into the program for inferencing.

If the `save_model_path` argument in `train_evaluate()` is set to a valid directory, the models will be saved locally. Specifically, the package saves `scikit_model.joblib` and `vectorizer.joblib` under the provided folder.

Please run `test_load.py` from the repository root directory to see the result:

```bash
python -m model5.package.test_load
```

### Sample Usage:

```python
from model5.package import load_saved_model

data = load_saved_model(
	saved_dir='model5/saved_model', 	# Required.
)
print(data)
```

### Notes:
	- You may access the models using `data['model']` and `data['embedding']`. They are in a model object type already, which mean you can directly use the models with their respective libraries. 
	- To utilize the loaded TF-IDF embedding model and scikit-learn Naive Bayes model, you may use the following:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer: TfidfVectorizer = data['embedding']
sample = vectorizer.transform(list_of_sample_texts)
model: NaiveBayesModel = data['model']
predictions = model.predict(sample)
```

	- A comprehensive example is provided in `test_load.py`, which also provides the classification prediction ability using scikit-learn libraries.

# Testing files if you are interested...
The folder `test-notebooks` contains the approaches that I have implemented in a Jupyter Notebook format. They are not used in the main implementation, but rather used for testing purpose. But if you are interested on how the TF-IDF and Naive Bayes mechanics work, you may check them out in that directory.