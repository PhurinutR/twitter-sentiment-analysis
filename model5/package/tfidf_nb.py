"""
Implementation of TF-IDF + Naive Bayes, training and evaluation.
"""

# Import all necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
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

def train_evaluate(train_csv: str, test_csv: str, use_spark: bool = False, save_model_path: Optional[str] = None, tfidf_params: Optional[dict] = None, nb_params: Optional[dict] = None):
    """
    Main function to fit TF-IDF on training data and train a Naive Bayes classifier, then evaluate it.

    Args:
        train_csv: Path to training CSV file. Required.
        test_csv: Path to testing CSV file. Required.
        use_spark: If True, use PySpark's `NaiveBayes`; otherwise use scikit-learn's `MultinomialNB`. Optional, default to False.
        save_model_path: If provided, save model and vectorizer objects to this directory. Optional, default to None.
        tfidf_params: Dict of params passed to `TfidfVectorizer`. Optional. If None, default params are used.
        nb_params: Dict of params passed to Naive Bayes model (scikit-learn's MultinomialNB or PySpark's NaiveBayes). Optional. If None, default params are used.
    
    Notes:
        - For TF-IDF, `tfidf_params` can include: 'min_df', 'max_df', 'ngram_range', 'use_idf', etc.
        - For PySpark NaiveBayes, `nb_params` can include: 'modelType' (default 'multinomial') and 'smoothing' (default 1.0).
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
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train = vectorizer.fit_transform(train_documents)
    X_test = vectorizer.transform(test_documents)

    # Dictionary to hold results
    result = {
        "train_accuracy": None,
        "test_accuracy": None,
        "train_log_loss": None,
        "test_log_loss": None,
    }

    # Use PySpark NaiveBayes
    if use_spark:
        print("\nUsing PySpark NaiveBayes...\n")
        from pyspark.sql import SparkSession
        from pyspark.ml.linalg import Vectors
        from pyspark.ml.classification import NaiveBayes
        from pyspark.sql.functions import udf, col
        from pyspark.sql.types import DoubleType
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator

        ss = SparkSession.builder.getOrCreate()

        # Convert sparse matrices to arrays for storing into Spark DataFrames
        X_train_arr = X_train.toarray()
        X_test_arr = X_test.toarray()

        # Build the Spark DataFrames
        train_rows = [(Vectors.dense(vec), int(lbl)) for vec, lbl in zip(X_train_arr, train_labels)]
        test_rows = [(Vectors.dense(vec), int(lbl)) for vec, lbl in zip(X_test_arr, test_labels)]
        col_name = "tf-idf vectors"
        col_label = "label"

        train_df = ss.createDataFrame(train_rows, [col_name, col_label])
        test_df = ss.createDataFrame(test_rows, [col_name, col_label])

        # Train Naive Bayes model
        nb_params_spark = {"featuresCol": col_name, "labelCol": col_label, "modelType": nb_params.get("modelType", "multinomial"), "smoothing": float(nb_params.get("smoothing", 1.0))}
        nb = NaiveBayes(**nb_params_spark)
        model = nb.fit(train_df)

        # Predictions and metrics
        def get_prob(probability, label):
            return float(probability[int(label)])
        get_prob_udf = udf(get_prob, DoubleType())

        train_prediction = model.transform(train_df).withColumn("true_prob", get_prob_udf(col("probability"), col(col_label)))
        test_pred = model.transform(test_df).withColumn("true_prob", get_prob_udf(col("probability"), col(col_label)))

        accuracy_evaluator = MulticlassClassificationEvaluator(labelCol=col_label, predictionCol="prediction", metricName="accuracy")
        loss_evaluator = MulticlassClassificationEvaluator(labelCol=col_label, predictionCol="prediction", metricName="logLoss")

        # Evaluation
        train_accuracy = accuracy_evaluator.evaluate(train_prediction)
        test_accuracy = accuracy_evaluator.evaluate(test_pred)
        train_loss = loss_evaluator.evaluate(train_prediction)
        test_loss = loss_evaluator.evaluate(test_pred)

        result["train_accuracy"] = float(train_accuracy * 100)
        result["test_accuracy"] = float(test_accuracy * 100)
        result["train_log_loss"] = float(train_loss)
        result["test_log_loss"] = float(test_loss)

        # NEWLY ADDED: Persist Spark model and vectorizer if requested
        if save_model_path:
            final_path = save_model_path + "/spark"
            os.makedirs(final_path, exist_ok=True)
            try:
                # save spark model using its write() API
                spark_model_path = os.path.join(final_path, "spark_model")
                model.write().overwrite().save(spark_model_path)
                vec_file = os.path.join(final_path, "vectorizer.pt")
                joblib.dump(vectorizer, vec_file)
            except Exception as e:
                print(f"saving spark model failed: {e}")

        ss.stop()
        return result
    
    # NEW ADDITION: Optionally use scikit-learn MultinomialNB if specified
    else:
        print("\nUsing Scikit-learn MultinomialNB...\n")
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score, log_loss

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
            model_file = os.path.join(final_path, "scikit_model.pt")
            vec_file = os.path.join(final_path, "vectorizer.pt")
            try:
                joblib.dump(model, model_file)
                joblib.dump(vectorizer, vec_file)
            except Exception as e:
                print(f"saving failed: {e}")

        return result

def load_saved_model(saved_dir: str):
    """Load a saved model and vectorizer locally.

    The function looks for the following structures under `saved_dir`:
    - `{saved_dir}/scikit/scikit_model.pt` and `{saved_dir}/scikit/vectorizer.pt` (scikit-learn)
    - `{saved_dir}/spark/spark_model` and `{saved_dir}/spark/vectorizer.pt` (PySpark)

    Args:
        saved_dir: Directory where the model and vectorizer are saved.

    Returns:
        dict: {
            'type': str,
            'model': object,
            'vectorizer': object,
            'paths': dict,
        }
    """

    saved_dir = os.path.abspath(saved_dir)
    result = {
        "type": "unknown", 
        "model": None, 
        "vectorizer": None, 
        "paths": {}
    }

    # Check scikit-learn path
    scikit_dir = os.path.join(saved_dir, "scikit")
    scikit_model_file = os.path.join(scikit_dir, "scikit_model.pt")
    scikit_vec_file = os.path.join(scikit_dir, "vectorizer.pt")

    if os.path.exists(scikit_model_file):
        try:
            model = joblib.load(scikit_model_file)
            vec = None
            if os.path.exists(scikit_vec_file):
                vec = joblib.load(scikit_vec_file)
            result.update({"type": "scikit", "model": model, "vectorizer": vec, "paths": {"model": scikit_model_file, "vectorizer": scikit_vec_file}})
            return result
        except Exception as e:
            print(f"failed to load scikit-learn files: {e}")
            return result

    # Check PySpark path
    spark_dir = os.path.join(saved_dir, "spark")
    spark_model_dir = os.path.join(spark_dir, "spark_model")
    spark_vec_file = os.path.join(spark_dir, "vectorizer.pt")

    if os.path.isdir(spark_model_dir):
        try:
            # Import here to avoid requiring pyspark if not used
            from pyspark.sql import SparkSession
            from pyspark.ml.classification import NaiveBayesModel

            ss = SparkSession.builder.getOrCreate()
            model = NaiveBayesModel.load(spark_model_dir)
            vec = None
            if os.path.exists(spark_vec_file):
                try:
                    vec = joblib.load(spark_vec_file)
                except Exception:
                    vec = None
            result.update({"type": "spark", "model": model, "vectorizer": vec, "paths": {"model": spark_model_dir, "vectorizer": spark_vec_file}})
            return result
        except Exception as e:
            print(f"failed to load spark model: {e}")
            return result

    print("No known model files found in provided directory")
    return None
