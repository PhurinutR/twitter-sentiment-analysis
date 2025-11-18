"""
Example usage of loading a model from the package in model 5.

To run this script from the repo root directory, use:
python -m model5.package.test_load
"""

import os
import sys
import pprint

# Import model5's package. If import fails, add root path to sys.path
try:
    from model5.package import evaluate_saved_model, load_saved_model
except Exception:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    from model5.package import evaluate_saved_model, load_saved_model

def main(): 
    # This is the same one in the train_model(). Modify these paths as needed
    save_dir = "model5/saved_models_pretrain7" 

    """
    Example usage of evaluating a saved TF-IDF embedding and Naive Bayes classifier model.
    """
    # Specify the training and testing data CSV files
    train_csv = "Twitter_data/pre_traindata7.csv"
    test_csv = "Twitter_data/testdata7.csv"

    # Play around with this function!
    eval = evaluate_saved_model(
        saved_dir=save_dir, 
        train_csv=train_csv, 
        test_csv=test_csv
    )
    print("\nEvaluation Results:")
    pprint.pprint(eval)

    """
    Example usage of loading a saved TF-IDF embedding and Naive Bayes classifier model.
    """
    # Play around with this function!
    data = load_saved_model(
        saved_dir=save_dir, 
    )
    print("\nLoading model Results:")
    pprint.pprint(data)
    print(type(data['embedding']))
    print(type(data['model']))

    """
    Example usage of using the loaded models, you can do something like this:
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer: TfidfVectorizer = data['embedding']

    from sklearn.naive_bayes import MultinomialNB
    model: MultinomialNB = data['model']

    sample_texts = ["I love this!", "I hate this!", "Sounds alright."]
    X_sample = vectorizer.transform(sample_texts)

    # Make predictions
    preds = model.predict(X_sample)
    print("\nPrediction Results:")
    for text, pred in zip(sample_texts, preds):
        print(f"Text: {text} => Predicted class: {pred}")   

if __name__ == '__main__':
    main()