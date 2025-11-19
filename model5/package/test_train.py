"""
Example usage of training and evaluation from the package in model 5.

To run this script from the repo root directory, use:
python -m model5.package.test_train
"""

import os
import sys
import pprint

# Import model 5's package. If import fails, add root path to sys.path
try:
    from model5.package import train_model
except Exception:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    from model5.package import train_model

def main(): 
    # Modify these paths as needed
    train_csv = "Twitter_data/traindata7.csv" 

    # Model saving directory (default saves to the model5/saved_models directory, set to None if you don't want to save models)
    # save_dir = "model5/saved_models"
    save_dir = "model5/saved_models"

    # Sample dictionary of parameters for TF-IDF and Naive Bayes
    tfidf_parameters = {"min_df": 4, "max_df": 0.95, "ngram_range": (1, 2)} # Fine-tuning of embedding model
    nb_parameters = {"alpha": 1.0}  # Example for scikit-learn MultinomialNB, fine-turning

    # playaround with this function!
    train_model(
        train_csv, 
        save_model_path=save_dir, 
        tfidf_params=tfidf_parameters, 
        nb_params=nb_parameters
    )

if __name__ == '__main__':
    main()
