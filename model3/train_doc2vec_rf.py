# This script handles training: preprocess data, train Doc2Vec, get embeddings, train RandomForest, hyperparameter tuning for both Doc2Vec and RandomForest

import os, sys, joblib

from pyspark.sql import SparkSession
from util.preprocessing import load_and_preprocess_data
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, Phraser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV

# Spark setup
spark = SparkSession.builder.appName("FinalTraining").getOrCreate()
spark.sparkContext.addPyFile(os.path.abspath('../util/preprocessing.py'))

# Load all three datasets (assuming all have 'Phrase' and 'Sentiment')
pre_train_df = load_and_preprocess_data("../Twitter_data/pre_traindata7.csv")
train_df      = load_and_preprocess_data("../Twitter_data/traindata7.csv")
test_df       = load_and_preprocess_data("../Twitter_data/testdata7.csv")

pre_train_pd = pre_train_df.toPandas()
train_pd      = train_df.toPandas()
test_pd       = test_df.toPandas()

print(f"Sizes → pre_train: {len(pre_train_pd)}, train: {len(train_pd)}, test: {len(test_pd)}")

# 1. Bigram detection on the largest set (pre_train + train)
all_sentences = pd.concat([pre_train_pd, train_pd])['Phrase'].str.split().tolist()
bigram = Phrases(all_sentences, min_count=5, threshold=10)
bigram_phraser = Phraser(bigram)

# 2. Prepare TaggedDocuments ONLY from pre_train (as per request)
pre_sentences = pre_train_pd['Phrase'].str.split().tolist()
tagged_docs = [
    TaggedDocument(bigram_phraser[sent], [i])
    for i, sent in enumerate(pre_sentences)
]

# 3. Hyper-parameter grid (much smaller + anti-overfit)
doc2vec_grid = [
    {'vector_size': 100, 'epochs': 30, 'dm': 0},
    {'vector_size': 150, 'epochs': 40, 'dm': 0},
    {'vector_size': 200, 'epochs': 30, 'dm': 0},
]

rf_grid = {
    'n_estimators': [50,100,200, 300],
    'max_depth': [8, 10, 12],
    'min_samples_leaf': [2, 4],
    'class_weight': ['balanced']
}

best_f1 = 0
best_d2v = None
best_rf_params = None

for params in doc2vec_grid:
    print(f"\n=== Doc2Vec {params} ===")
    d2v = Doc2Vec(
        vector_size=params['vector_size'],
        epochs=params['epochs'],
        dm=params['dm'],
        window=8, min_count=2, workers=6,
        alpha=0.025, min_alpha=0.0001
    )
    d2v.build_vocab(tagged_docs)
    d2v.train(tagged_docs, total_examples=d2v.corpus_count, epochs=d2v.epochs)

    # Embeddings for pretrain set (large, labeled)
    pre_emb = np.array([
        d2v.infer_vector(bigram_phraser[row['Phrase'].split()])
        for _, row in pre_train_pd.iterrows()
    ])
    y_pre = pre_train_pd['Sentiment'].values

    # CV on pretrain set for hyperparameter tuning
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_grid, cv=5, scoring='f1_macro', n_jobs=-1
    )
    gs.fit(pre_emb, y_pre)

    if gs.best_score_ > best_f1:
        best_f1 = gs.best_score_
        best_d2v = d2v
        best_rf_params = gs.best_params_
        print(f"New best CV macro-F1 on pretrain: {best_f1:.4f}")

print(f"\nFinal best CV macro-F1 on pretrain: {best_f1:.4f}")

# With best Doc2Vec, infer embeddings for small train and test
train_emb = np.array([
    best_d2v.infer_vector(bigram_phraser[row['Phrase'].split()])
    for _, row in train_pd.iterrows()
])
y_train = train_pd['Sentiment'].values

test_emb = np.array([
    best_d2v.infer_vector(bigram_phraser[row['Phrase'].split()])
    for _, row in test_pd.iterrows()
])

# Pre_emb already from best_d2v (from last loop, but to be sure, re-infer if needed; but since last is not necessarily best, actually need to re-infer)
# Wait, to fix: after loop, re-compute pre_emb with best_d2v
pre_emb = np.array([
    best_d2v.infer_vector(bigram_phraser[row['Phrase'].split()])
    for _, row in pre_train_pd.iterrows()
])

# 4. Pretrain RF: Create and fit RF with best params on pretrain data
print("\n=== Pretraining RF on Pretrain Data ===")
clf = RandomForestClassifier(**best_rf_params, random_state=42)
clf.fit(pre_emb, y_pre)

# 5. Finetune: Retrain on combined data with higher weights on small train
print("\n=== Finetuning RF on Combined (emphasis on small) ===")
combined_emb = np.vstack([pre_emb, train_emb])
y_combined = np.concatenate([y_pre, y_train])

# Weigh small data higher (e.g., 5x; adjust based on sizes/imbalance)
sample_weights = np.ones(len(y_combined))
sample_weights[len(y_pre):] *= 5

clf.fit(combined_emb, y_combined, sample_weight=sample_weights)  # Overwrite with finetuned fit

# Save everything
os.makedirs("saved_models", exist_ok=True)
best_d2v.save("saved_models/best_doc2vec.model")
joblib.dump(clf, "saved_models/best_classifier.pkl")
bigram_phraser.save("saved_models/bigram_phraser.model")

# Final test performance (real held-out test)
pred = clf.predict(test_emb)
print("\n=== FINAL TEST PERFORMANCE ===")
print(classification_report(test_pd['Sentiment'], pred))
print("Test macro-F1:", f1_score(test_pd['Sentiment'], pred, average='macro'))