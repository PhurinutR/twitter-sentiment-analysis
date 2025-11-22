# Load the saved Decision Tree sentiment model + CountVectorizer

import os
from model4.package.decision_tree_model import load_saved_model
from model4.package.preprocess import clean_text

SAVE_DIR = "saved_model"

# Load model + vectorizer
loaded = load_saved_model(SAVE_DIR)
model = loaded["model"]
vectorizer = loaded["embedding"]

print("Model loaded successfully!")
print("Vectorizer vocabulary size:", len(vectorizer.vocabulary_))

# Test on new text
sample_text = "I really enjoyed this movie, it was fantastic!"
sample_text_cleaned = clean_text(sample_text)

prediction = model.predict([sample_text_cleaned])[0]
print("Prediction:", prediction)
