# train_word2vec_rf_tuned.py ← BEST VERSION WITH GRID SEARCH + CLEAR BEST PARAMS OUTPUT
import os
import joblib
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score
from pyspark.sql import SparkSession
from util.preprocessing import load_and_preprocess_data

# Spark setup
spark = SparkSession.builder.appName("Word2VecTuning").getOrCreate()
spark.sparkContext.addPyFile(os.path.abspath('../util/preprocessing.py'))

# Load data
pre_train_df = load_and_preprocess_data("../Twitter_data/pre_traindata7.csv")
train_df = load_and_preprocess_data("../Twitter_data/traindata7.csv")
test_df = load_and_preprocess_data("../Twitter_data/testdata7.csv")

pre_train_pd = pre_train_df.toPandas()
train_pd = train_df.toPandas()
test_pd = test_df.toPandas()

print(f"Sizes → pre_train: {len(pre_train_pd):,}, train: {len(train_pd):,}, test: {len(test_pd):,}")

# 1. Bigram detection
all_sentences = pd.concat([pre_train_pd, train_pd])['Phrase'].str.split().tolist()
bigram = Phrases(all_sentences, min_count=5, threshold=10)
bigram_phraser = Phraser(bigram)

# Apply bigrams
pre_train_phrased = [bigram_phraser[sent] for sent in pre_train_pd['Phrase'].str.split()]
train_phrased = [bigram_phraser[sent] for sent in train_pd['Phrase'].str.split()]
test_phrased = [bigram_phraser[sent] for sent in test_pd['Phrase'].str.split()]

all_phrased = pre_train_phrased + train_phrased

# 2. Word2Vec hyperparameter grid
w2v_params_grid = [
    {'vector_size': 150, 'window': 8,  'sg': 1, 'epochs': 30, 'negative': 10},
    {'vector_size': 200, 'window': 8,  'sg': 1, 'epochs': 40, 'negative': 15},
    {'vector_size': 200, 'window': 10, 'sg': 1, 'epochs': 30, 'negative': 10},
    {'vector_size': 300, 'window': 8,  'sg': 1, 'epochs': 30, 'negative': 10},
]

# 3. Random Forest grid
rf_param_grid = {
    'n_estimators': [50,100,200, 300],
    'max_depth': [8, 10, 12],
    'min_samples_leaf': [2, 4],
    'class_weight': ['balanced'],
}

best_macro_f1 = 0
best_w2v_model = None
best_rf_model = None
best_params = {}

print("\n" + "="*70)
print("STARTING GRID SEARCH: Word2Vec + RandomForest")
print("="*70)

for i, w2v_params in enumerate(w2v_params_grid, 1):
    print(f"\n[{i}/{len(w2v_params_grid)}] Training Word2Vec with:")
    for k, v in w2v_params.items():
        print(f"   • {k:12}: {v}")
    
    w2v = Word2Vec(
        sentences=all_phrased,
        vector_size=w2v_params['vector_size'],
        window=w2v_params['window'],
        sg=w2v_params['sg'],
        epochs=w2v_params['epochs'],
        negative=w2v_params['negative'],
        min_count=3,
        workers=6,
        alpha=0.03,
        min_alpha=0.0007,
        seed=42,
        sample=1e-3,
        hs=0
    )

    def get_embedding(tokens):
        vecs = [w2v.wv[word] for word in tokens if word in w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size)

    pre_emb = np.array([get_embedding(tokens) for tokens in pre_train_phrased])
    y_pre = pre_train_pd['Sentiment'].values

    # Grid search RF
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=rf_param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(pre_emb, y_pre)

    current_f1 = grid_search.best_score_
    print(f"   → Best CV macro-F1: {current_f1:.4f} | RF params: {grid_search.best_params_}")

    if current_f1 > best_macro_f1:
        best_macro_f1 = current_f1
        best_w2v_model = w2v
        best_rf_model = grid_search.best_estimator_
        best_params = {
            'word2vec': w2v_params.copy(),
            'random_forest': grid_search.best_params_
        }
        print(f"   NEW BEST MODEL! macro-F1 = {best_macro_f1:.4f}")
        print("   Updated best parameters!")

# === FINAL MODEL TRAINING ===
print("\n" + "="*70)
print("FINAL TRAINING WITH BEST HYPERPARAMETERS")
print("="*70)

def final_embedding(tokens):
    vecs = [best_w2v_model.wv[word] for word in tokens if word in best_w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(best_w2v_model.vector_size)

train_emb = np.array([final_embedding(t) for t in train_phrased])
test_emb  = np.array([final_embedding(t) for t in test_phrased])
pre_emb   = np.array([final_embedding(t) for t in pre_train_phrased])

y_train = train_pd['Sentiment'].values
y_test  = test_pd['Sentiment'].values
y_pre   = pre_train_pd['Sentiment'].values

# Combine pre_train + train with weight on real labeled data
combined_emb = np.vstack([pre_emb, train_emb])
y_combined   = np.concatenate([y_pre, y_train])
sample_weights = np.ones(len(y_combined))
sample_weights[len(y_pre):] *= 5  # Emphasize actual labeled train data

print(f"Final training on {len(y_combined):,} samples (weighted)")

best_rf_model.fit(combined_emb, y_combined, sample_weight=sample_weights)

# Save models
os.makedirs("saved_models2", exist_ok=True)
best_w2v_model.save("saved_models2/word2vec_best.model")
joblib.dump(best_rf_model, "saved_models2/best_classifier.pkl")
bigram_phraser.save("saved_models2/bigram_phraser.model")

# Final evaluation
pred = best_rf_model.predict(test_emb)
test_acc = np.mean(pred == y_test)
test_f1  = f1_score(y_test, pred, average='macro')

print("\n" + "="*70)
print("FINAL TEST RESULTS")
print("="*70)
print(classification_report(y_test, pred))
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Test macro-F1  : {test_f1:.4f}")

# BEST HYPERPARAMETERS - CLEARLY DISPLAYED
print("\n" + "="*70)
print("BEST HYPERPARAMETERS FOUND")
print("="*70)
print("Word2Vec Parameters:")
for k, v in best_params['word2vec'].items():
    print(f"   • {k:12}: {v}")
print("\nRandom Forest Parameters:")
for k, v in best_params['random_forest'].items():
    print(f"   • {k:12}: {v}")
print(f"\nBest CV macro-F1 : {best_macro_f1:.4f}")
print(f"Final Test macro-F1 : {test_f1:.4f}")
print("="*70)
