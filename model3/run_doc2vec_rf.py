# model3/run_final.py
import os, sys, joblib, pandas as pd, numpy as np


from pyspark.sql import SparkSession
from util.preprocessing import load_and_preprocess_data
from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phraser

spark = SparkSession.builder.appName("Inference").getOrCreate()
spark.sparkContext.addPyFile(os.path.abspath('../util/preprocessing.py'))

# Load real test data
test_df = load_and_preprocess_data("../Twitter_data/testdata7.csv")
test_pd = test_df.toPandas()

# Load everything
doc2vec = Doc2Vec.load("saved_models/best_doc2vec.model")
clf     = joblib.load("saved_models/best_classifier.pkl")
phraser = Phraser.load("saved_models/bigram_phraser.model")

# Predict
emb = np.array([
    doc2vec.infer_vector(phraser[row['Phrase'].split()])
    for _, row in test_pd.iterrows()
])
pred = clf.predict(emb)

print("Final test accuracy:", (pred == test_pd['Sentiment']).mean())
#print(classification_report(test_pd['Sentiment'], pred))