""" 
Implementation of TF-IDF + Naive Bayes, training and evaluation.
"""

# Import all necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, log_loss
from typing import Optional
import os, sys, joblib

# Set PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON to current Python executable, 
# to prevent PySpark from using a different Python interpreter
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

try:
    from util.preprocessing import load_and_preprocess_data
except Exception:
    # Adding the root path to the sys context for the runtime to properly import util.preprocessing.
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    from util.preprocessing import load_and_preprocess_data

# Functions
def extract_docs_labels(spark_df, text_col: str = "Phrase", label_col: str = "Sentiment"):
    """
    Parent function to convert Spark DataFrame to lists of documents and labels (strings and ints).
    """
    print("\nExtracting docs and labels from data...")
    pdf = spark_df.select(text_col, label_col).toPandas()
    documents = pdf.iloc[:, 0].astype(str).tolist()
    labels = pdf.iloc[:, 1].astype(int).tolist()
    return documents, labels

def train_evaluate(train_csv: str, test_csv: str, save_model_path: Optional[str] = None, tfidf_params: Optional[dict] = None, nb_params: Optional[dict] = None):
    """
    Main function to fit TF-IDF on training data and train a Naive Bayes classifier, then evaluate it.

    Args:
        train_csv: Path to training CSV file. Required.
        test_csv: Path to testing CSV file. Required.
        save_model_path: If provided, save model and vectorizer objects to this directory. Optional, default to None.
        tfidf_params: Dict of params passed to `TfidfVectorizer`. Optional. If None, default params are used.
        nb_params: Dict of params passed to Naive Bayes model (scikit-learn's MultinomialNB). Optional. If None, default params are used.
    
    Notes:
        - For TF-IDF, `tfidf_params` can include: 'min_df', 'max_df', 'ngram_range', 'use_idf', etc.
        - For scikit-learn MultinomialNB, `nb_params` can include: any valid parameters for `MultinomialNB`.
        - To save models, ensure `save_model_path` is set to a valid directory path.

    Returns:
        dict: {
            'train_accuracy': float,
            'test_accuracy': float,
            'train_log_loss': float,
            'test_log_loss': float,
        }

    """
    print("\nStarted TD-IDF + Naive Bayes pipeline...\n")

    # Parameters
    text_column = "Phrase"
    sentiment_column = "Sentiment"
    tfidf_params = tfidf_params or {"min_df": 4, "max_df": 0.95, "ngram_range": (1, 2), "use_idf": True}
    nb_params = nb_params or {}

    # Load and preprocess using util package
    train_spark_df = load_and_preprocess_data(train_csv, text_column, sentiment_column)
    test_spark_df = load_and_preprocess_data(test_csv, text_column, sentiment_column)

    # Convert to lists for scikit path
    train_documents, train_labels = extract_docs_labels(train_spark_df, text_column, sentiment_column)
    test_documents, test_labels = extract_docs_labels(test_spark_df, text_column, sentiment_column)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(**tfidf_params)
    X_train = tfidf.fit_transform(train_documents)
    X_test = tfidf.transform(test_documents)

    # Dictionary to hold results
    result = {
        "train_accuracy": None,
        "test_accuracy": None,
        "train_log_loss": None,
        "test_log_loss": None,
    }

    # Use scikit-learn MultinomialNB (works with sparse matrices)
    model = MultinomialNB(**nb_params)
    model.fit(X_train, train_labels)

    # Predict and compute metrics
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    # Probabilities and metrics
    train_probability = model.predict_proba(X_train)
    test_probability = model.predict_proba(X_test)

    # Evaluation
    result["train_accuracy"] = float(accuracy_score(train_labels, train_prediction) * 100)
    result["test_accuracy"] = float(accuracy_score(test_labels, test_prediction) * 100)
    result["train_log_loss"] = float(log_loss(train_labels, train_probability, labels=model.classes_))
    result["test_log_loss"] = float(log_loss(test_labels, test_probability, labels=model.classes_))

    # NEWLY ADDED: Persist model and vectorizer if requested
    if save_model_path:
        final_path = save_model_path + "/scikit"
        os.makedirs(final_path, exist_ok=True)
        try:
            # 1. save model using joblib
            model_file = os.path.join(final_path, "scikit_model.pt")
            joblib.dump(model, model_file)
            # 2. save embedding model using joblib
            vec_file = os.path.join(final_path, "embedding.pt")
            joblib.dump(tfidf, vec_file)
        except Exception as e:
            print(f"saving failed: {e}")

    return result

def load_saved_model(saved_dir: str):
    """Load a saved embedding and classifier models locally.

    The function looks for the following structures under `saved_dir`:
    - `{saved_dir}/scikit/scikit_model.pt` and `{saved_dir}/scikit/embedding.pt` (scikit-learn)

    Args:
        saved_dir: Directory where the TF-IDF embedding and Naive Bayes classifier models are saved.

    Returns:
        dict: {
            'embedding': object,
            'model': object,
        }
    """

    saved_dir = os.path.abspath(saved_dir)
    result = {
        "model": None, 
        "embedding": None, 
    }

    # Check scikit-learn path
    scikit_dir = os.path.join(saved_dir, "scikit")
    scikit_model_file = os.path.join(scikit_dir, "scikit_model.pt")
    scikit_vec_file = os.path.join(scikit_dir, "embedding.pt")

    model, embed = None, None

    if os.path.exists(scikit_model_file):
        try:
            model = joblib.load(scikit_model_file)
            if os.path.exists(scikit_vec_file):
                embed = joblib.load(scikit_vec_file)
            result.update({"embedding": embed, "model": model})
            return result
        except Exception as e:
            print(f"failed to load scikit-learn files: {e}")
            return result

    print("No known model files found in provided directory")
    return None