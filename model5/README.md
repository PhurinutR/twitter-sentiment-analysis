# Model 5 Implementation Guide
Core architecture: **TF-IDF** embedding model with **Naive Bayes** classifier model<br>
Author: *Marcus KWAN TH*<br>

## Prerequisite
1. Clone this GitHub repository into the local PC.
2. Run the commands provided in the "Getting Started" guide in the home page to install necessary libraries.

## Using the TF-IDF with Naive Bayes Package for **Training and Evaluation**

A function to run TF-IDF + Naive Bayes and return the accuracy with loss metrics are provided by `train_evaluate()`, within a package `tfidf_nb.py` under `model5/package`.

### Procedures:
1. Please run `test_train.py` from the repository root directory:

```bash
python -m model5.package.test_train
```

2. Sample Usage:

```python
from model5.package import train_evaluate

res = train_evaluate(
	'Twitter_data/traindata.csv',
	'Twitter_data/testdata.csv',
    tfidf_parameters = {"min_df": 4, "max_df": 0.95, "ngram_range": (1, 2), "use_idf": True} # Optional: Sample paramters for TF-IDF embedding
    nb_parameters = {"alpha": 1.0}  # Optional: Sample parameters for scikit-learn's Naive Bayes classifier
	save_model_path='model5/package/saved_models'  # Optional: persist embedding model and classifier model
)
print(res)
```

3. Notes:
- The package is run from the root directory so the `util` package is importable.
- Some verbose and warning from PySpark may appear in the console output. They are normal and no need to be worried about.

## Using the TF-IDF with Naive Bayes Package for **Loading the Saved Model and Prediction**

Another function in `tfidf_nb.py` is `load_saved_model()`. As the name suggest, it loads the saved model from a directory into the program for inferencing.

If the `save_model_path` argument in `train_evaluate()` is set to a valid directory, the models will be saved locally. Specifically, the package saves `scikit_model.joblib` and `vectorizer.joblib` under the provided folder.

### Procedures:
1. You may run the file `test_load.py` from the repo root directory to test the model-loading function:

```bash
python -m model5.package.test_load
```

2. Sample Usage:

```python
from model5.package import load_saved_model

data = load_saved_model(
	saved_dir='model5/saved_model', 
)
print(data)
```

3. You may access the models using `data['model']` and `data['embedding']`. They are in a model object type already, which mean you can directly use the models with their respective libraries. To load TF-IDF embedding model and scikit-learn Naive Bayes model, you may use the following:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer: TfidfVectorizer = data['embedding']
sample = vectorizer.transform(list_of_sample_texts)
model: NaiveBayesModel = data['model']
predictions = model.predict(sample)
```

A comprehensive example is provided in `test_load.py`, which also provides the classification prediction ability using scikit-learn libraries.

# Testing files if you are interested...
This part simply documented the approaches that I have implemented in a Notebook format. They are now used for testing purpose, but if you are interested on how the TF-IDF and Naive Bayes mechanics work, you may check them out in the `model5/Test Notebooks` directory.
## Using TF-IDF with Naive Bayes Jupyter Notebook
⚠️ Note: Before running the notebook, please ensure the file `util.zip` is present in the root directory. If not, zip the util folder using the following command: 

```bash
cd twitter-sentiment-analysis
zip -r util.zip util
```

This step is necessary for PySpark to access the util package during distributed processing in the Notebook files.

1. Run `tfidf_nb_program.ipynb` from the `Test Notebook` folder.
2. Follow the instructions inside the Jupyter Notebook to play around with it.
3. At the end, you will obtain a train/testing loss and accuracy score, fine-tune the parameters for `TfidfVectorizer()` and `NaiveBayes()` to adjust the parameters.

## TF-IDF Word-embedding Exporter Jupyter Notebook
1. Run `tfidf_word_embed.ipynb` from the `Test Notebook` folder.
2. Follow the instructions inside the Jupyter Notebook to play around with it.
3. You may check the `output` folder for the exported TF-IDF vectors after running the last cell of the file.