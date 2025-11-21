# model3/run_final.py
import os, sys, joblib, pandas as pd, numpy as np
from pyspark.sql import SparkSession
from util.preprocessing import load_and_preprocess_data
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser

spark = SparkSession.builder.appName("Inference").getOrCreate()
spark.sparkContext.addPyFile(os.path.abspath('../util/preprocessing.py'))

# Load real test data
test_df = load_and_preprocess_data("../Twitter_data/testdata7.csv")
test_pd = test_df.toPandas()

# Load everything
w2v = Word2Vec.load("saved_models2/word2vec_best.model")
clf = joblib.load("saved_models2/best_classifier.pkl")
phraser = Phraser.load("saved_models2/bigram_phraser.model")



#to load the model directly from hf (comment the code above out):
# repo_id = "chrislhg/word2vec-rf"

# # Download model files
# w2v_path = hf_hub_download(repo_id=repo_id, filename="word2vec_best.model")
# clf_path = hf_hub_download(repo_id=repo_id, filename="best_classifier.pkl")
# phraser_path = hf_hub_download(repo_id=repo_id, filename="bigram_phraser.model")

# # Load models
# w2v = Word2Vec.load(w2v_path)
# clf = joblib.load(clf_path)
# phraser = Phraser.load(phraser_path)

# Predict
emb = np.array([
    np.mean([w2v.wv[t] for t in phraser[row['Phrase'].split()] if t in w2v.wv], axis=0)
    if any(t in w2v.wv for t in phraser[row['Phrase'].split()])
    else np.zeros(w2v.vector_size)
    for _, row in test_pd.iterrows()
])
pred = clf.predict(emb)

print("Final test accuracy:", (pred == test_pd['Sentiment']).mean())
