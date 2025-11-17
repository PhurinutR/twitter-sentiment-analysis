# Model 5 Implementation
Core architecture: **TF-IDF** embedding model with **Naive Bayes** classifier model<br>
Author: *Marcus KWAN TH*<br>

## Prerequisite:
1. Clone this GitHub repository into the local PC.
2. Run the commands provided in the "Getting Started" guide in the home page.

## Using the TFIDF with Naive Bayes Package for Training and Evaluation

A small package `tfidf_nb.py` is provided at `model5/package` to run TF-IDF + Naive Bayes and return accuracy/loss metrics.

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
	use_spark=True,                 # default; set False for scikit-learn one
    tfidf_parameters = {"min_df": 5, "max_df": 0.9, "ngram_range": (1, 2), "use_idf": True}     # Optional: Sample paramters for TF-IDF embedding
    nb_parameters = {"modelType": "multinomial", "smoothing": 1.0}  # Optional: Sample parameters for PySpark NaiveBayes
	save_model_path='model5/package/saved_models/scikit'  # optional: persist model+vectorizer
)
print(res)
```

3. Saved files:
- If the `save_model_path` argument is set to a valid directory, the models will be saved locally.
	- For scikit-learn path the package saves `scikit_model.joblib` and `vectorizer.joblib` under the provided folder.
	- For the PySpark path the package saves the Spark model directory `spark_model` and the `vectorizer.joblib` file.
- You may run the file `test_load.py` to test the model-loading function:
```bash
python -m model5.package.test_train
```

4. Notes:
- The package is run from the root directory so the `util` package is importable.
- Some verbose and warning from PySpark may appear in the console output. They are normal and no need to be worried about.
- If you found that PySpark approach runs slow in the program, consider switching to scikit-learn, which are much more efficient. They should works the same way regardless. (I just include both of them for the sake of the project)

# Testing files if you are interested...
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