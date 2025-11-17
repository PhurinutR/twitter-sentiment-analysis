
# Twitter Sentiment Analysis with Doc2Vec + Random Forest

A complete pipeline for Twitter/X sentiment classification using **Doc2Vec** for document embeddings and **Random Forest** as the classifier. The project leverages a large labeled pre-training set (~59k tweets) and a smaller high-quality training set (~600 tweets) to achieve robust performance.

## Overview

1. **Bigram detection** with Gensim Phrases → better phrase handling  
2. **Doc2Vec** trained on the large pre-train corpus (unsupervised + supervised)  
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

### 1. Train the Model

```bash
python model3/train_doc2vec_rf.py
```

- Trains Doc2Vec on the large pre-train data  
- Performs hyperparameter search for both Doc2Vec and Random Forest  
- Pre-trains RF on the large dataset  
- Finetunes RF on large + small data (small data weighted higher)  
- Saves everything to `saved_models/`  
- Prints final test macro-F1 and classification report

### 2. Run Inference on the Test Set

```bash
python model3/run_doc2vec_rf.py
```

Loads the saved models and evaluates on `testdata7.csv`.  
Uncomment the last lines if you want the full classification report.

### 3. Use the Saved Model in Your Own Code

```python
import joblib
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phraser

# Load models
doc2vec = Doc2Vec.load("saved_models/best_doc2vec.model")
clf     = joblib.load("saved_models/best_classifier.pkl")
phraser = Phraser.load("saved_models/bigram_phraser.model")

# Single prediction
text = "I love this movie so much!"
tokens = phraser[text.split()]
vector = doc2vec.infer_vector(tokens)
pred   = clf.predict(np.array([vector]))[0]
print("Predicted sentiment:", pred)

# Batch prediction
new_df = pd.DataFrame({"Phrase": ["great day", "terrible service", "ok I guess"]})
embeddings = np.array([
    doc2vec.infer_vector(phraser[row["Phrase"].split()])
    for _, row in new_df.iterrows()
])
new_df["Predicted_Sentiment"] = clf.predict(embeddings)
print(new_df)
```

