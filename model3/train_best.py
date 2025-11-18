# This script trains the final model using the best hyperparameters found
# It uses pre_traindata7.csv for Doc2Vec training and bigram detection
# Trains RF on pre_train, then finetunes on combined (pre_train + traindata7.csv) with weights
# Evaluates on testdata7.csv and saves models

import os
import sys
import joblib
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, Phraser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from pyspark.sql import SparkSession
from util.preprocessing import load_and_preprocess_data

# Spark setup (if PySpark is available; otherwise replace with Pandas loading below)
spark = SparkSession.builder.appName("FinalModelTraining").getOrCreate()
spark.sparkContext.addPyFile(os.path.abspath('../util/preprocessing.py'))

# Load all three datasets
pre_train_df = load_and_preprocess_data("../Twitter_data/pre_traindata7.csv")
train_df      = load_and_preprocess_data("../Twitter_data/traindata7.csv")
test_df       = load_and_preprocess_data("../Twitter_data/testdata7.csv")

pre_train_pd = pre_train_df.toPandas()
train_pd      = train_df.toPandas()
test_pd       = test_df.toPandas()

print(f"Sizes → pre_train: {len(pre_train_pd)}, train: {len(train_pd)}, test: {len(test_pd)}")

# 1. Bigram detection on pre_train + train
combined_df = pd.concat([pre_train_pd, train_pd], ignore_index=True)
all_sentences = combined_df['Phrase'].str.split().tolist()
bigram = Phrases(all_sentences, min_count=5, threshold=10)
bigram_phraser = Phraser(bigram)

# 2. Prepare TaggedDocuments from pre_train
pre_sentences = pre_train_pd['Phrase'].str.split().tolist()
tagged_docs = [TaggedDocument(bigram_phraser[sent], [i]) for i, sent in enumerate(pre_sentences)]

# 3. Train Doc2Vec with best hyperparameters
print("\n=== Training Doc2Vec with Best Params ===")
best_d2v_params = {
    'vector_size': 150,
    'epochs': 40,
    'dm': 0,
    'window': 8,
    'min_count': 2,
    'workers': 6,  # Adjust based on your CPU cores
    'alpha': 0.025,
    'min_alpha': 0.0001
}
doc2vec_model = Doc2Vec(**best_d2v_params)
doc2vec_model.build_vocab(tagged_docs)
doc2vec_model.train(tagged_docs, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# 4. Infer embeddings for all sets
print("\n=== Inferring Embeddings ===")
def infer_embeddings(df):
    return np.array([
        doc2vec_model.infer_vector(bigram_phraser[row['Phrase'].split()])
        for _, row in df.iterrows()
    ])

pre_emb = infer_embeddings(pre_train_pd)
train_emb = infer_embeddings(train_pd)
test_emb = infer_embeddings(test_pd)

y_pre = pre_train_pd['Sentiment'].values
y_train = train_pd['Sentiment'].values
y_test = test_pd['Sentiment'].values

# 5. Best RF params
best_rf_params = {
    'n_estimators': 300,
    'max_depth': 12,
    'min_samples_leaf': 4,
    'class_weight': 'balanced',
    'random_state': 42
}

# 6. Pretrain RF on pre_train data
print("\n=== Pretraining RF on Pretrain Data ===")
clf = RandomForestClassifier(**best_rf_params)
clf.fit(pre_emb, y_pre)

# 7. Finetune on combined with higher weights on small train
print("\n=== Finetuning RF on Combined Data ===")
combined_emb = np.vstack([pre_emb, train_emb])
y_combined = np.concatenate([y_pre, y_train])

sample_weights = np.ones(len(y_combined))
sample_weights[len(y_pre):] *= 5  # Adjust multiplier if needed

clf.fit(combined_emb, y_combined, sample_weight=sample_weights)

# 8. Save models
os.makedirs("saved_models", exist_ok=True)
doc2vec_model.save("saved_models/best_doc2vec.model")
joblib.dump(clf, "saved_models/best_classifier.pkl")
bigram_phraser.save("saved_models/bigram_phraser.model")

# 9. Final test performance
pred = clf.predict(test_emb)
print("\n=== FINAL TEST PERFORMANCE ===")
print(classification_report(y_test, pred))
print("Test macro-F1:", f1_score(y_test, pred, average='macro'))
