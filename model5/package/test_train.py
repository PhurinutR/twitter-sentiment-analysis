"""
Example usage of training and evaluation from the package in model 5.

To run this script from the repo root directory, use:
python -m model5.package.test_train
"""

import os
import sys
import pprint

# Import util.preprocessing. If import fails, add root path to sys.path
try:
    from model5.package import train_evaluate
except Exception:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    from model5.package import train_evaluate

def main():
    # Set to False to use scikit-learn instead of PySpark
    spark = False

    # Modify these paths as needed
    train_csv = "Twitter_data/pre_traindata7.csv"
    test_csv = "Twitter_data/testdata7.csv"

    # Model saving directory (default saves to the model5/saved_models directory, set to None if you don't want to save models)
    save_dir = "model5/saved_models"

    # Sample dictionary of parameters for TF-IDF and Naive Bayes
    tfidf_parameters = {"min_df": 4, "max_df": 0.95, "ngram_range": (1, 2)}
    nb_parameters = {"smoothing": 1.0}  # Example for PySpark NaiveBayes
    if not spark:
        nb_parameters = {"alpha": 1.0}  # Example for scikit-learn MultinomialNB

    # playaround with this function!
    result = train_evaluate(
        train_csv, 
        test_csv, 
        use_spark=spark, 
        save_model_path=save_dir, 
        tfidf_params=tfidf_parameters, 
        nb_params=nb_parameters
    )

    pprint.pprint(result)

if __name__ == '__main__':
    main()
