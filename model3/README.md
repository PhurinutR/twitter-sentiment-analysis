# Twitter Sentiment Analysis with Word2Vec + Random Forest

A complete pipeline for Twitter/X sentiment classification using **Word2Vec** for word embeddings (averaged into sentence vectors) and **Random Forest** as the classifier. The project leverages a large labeled pre-training set (~59k tweets) and a smaller high-quality training set (~600 tweets) to achieve robust performance.

## Overview

1. **Bigram detection** with Gensim Phrases → better phrase handling  
2. **Word2Vec** trained on the large pre-train corpus (unsupervised + supervised)  
3. **Random Forest** pretrained on the large labeled data → stable hyperparameters  
4. **Finetuning** of the Random Forest on a combination of large + small data (with higher weight on the small set)  
5. Models saved for easy inference

## Requirements

```txt
pyspark
pandas
numpy
gensim
scikit-learn
joblib
```

## Usage

### 1. Train the Model (for usage, skip step 1,2)

```bash
python train_word2vec_rf_tuned.py
```

- Trains Word2Vec on the large pre-train data  
- Performs hyperparameter search for both Word2Vec and Random Forest  
- Pre-trains RF on the large dataset  
- Finetunes RF on large + small data (small data weighted higher)  
- Saves everything to `saved_models2/`  
- Prints final test macro-F1 and classification report

### 2. Run Inference on the Test Set

```bash
python model3/run_final.py
```

Loads the saved models and evaluates on `testdata7.csv`.  
Uncomment the last lines if you want the full classification report.

### 3. Load Pre-trained Model from Hugging Face

You can download and load the pre-trained models directly from the Hugging Face Hub repository at [https://huggingface.co/chrislhg/word2vec-rf](https://huggingface.co/chrislhg/word2vec-rf). This allows you to use the model without training it from scratch.

First, install the `huggingface_hub` library if you haven't already:

```bash
pip install huggingface_hub
```

Then, use the following code to download and load the models:

```python
from huggingface_hub import hf_hub_download
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
import joblib

repo_id = "chrislhg/word2vec-rf"

# Download model files
w2v_path = hf_hub_download(repo_id=repo_id, filename="word2vec_best.model")
clf_path = hf_hub_download(repo_id=repo_id, filename="best_classifier.pkl")
phraser_path = hf_hub_download(repo_id=repo_id, filename="bigram_phraser.model")

# Load models
w2v = Word2Vec.load(w2v_path)
clf = joblib.load(clf_path)
phraser = Phraser.load(phraser_path)
```

### 4. Use the Saved Model in Your Own Code

```python
import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser

# Load models
w2v = Word2Vec.load("saved_models2/word2vec_best.model")
clf = joblib.load("saved_models2/best_classifier.pkl")
phraser = Phraser.load("saved_models2/bigram_phraser.model")

# Single prediction
text = "I love this movie so much!"
tokens = phraser[text.split()]
vecs = [w2v.wv[word] for word in tokens if word in w2v.wv]
vector = np.mean(vecs, axis=0) if vecs else np.zeros(w2v.vector_size)
pred = clf.predict(np.array([vector]))[0]
print("Predicted sentiment:", pred)

# Batch prediction
new_df = pd.DataFrame({"Phrase": ["great day", "terrible service", "ok I guess"]})
embeddings = np.array([
    np.mean([w2v.wv[t] for t in phraser[row["Phrase"].split()] if t in w2v.wv], axis=0)
    if any(t in w2v.wv for t in phraser[row["Phrase"].split()])
    else np.zeros(w2v.vector_size)
    for _, row in new_df.iterrows()
])
new_df["Predicted_Sentiment"] = clf.predict(embeddings)
print(new_df)
```
