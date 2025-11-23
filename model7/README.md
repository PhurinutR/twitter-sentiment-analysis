# Model7 RNN Implementation

This folder provides a Python package that encapsulates the GloVe embedding, RNN training, and inference workflow.

The main package file: `rnn_package/model.py` contains the core architecture of the implemetation, the details as follows:
  - `train_rnn(data_dir, run_dir, **kwargs)`: trains the model with configurable hyperparameters via `**kwargs`, saves model data and Fields, and returns accuracy and loss metrics.
  - `predict_texts(text_list)`: loads saved best model and reconstructs the model ready for, classify list of texts data into sentiments, then obtain the predictions.


## Prerequisite

1. Make sure to install all necessary libraries by following the **"Getting Started"** guide in the repo home page.
2. Pre-process the training data first
3. run python -m spacy download en_core_web_sm and python -m spacy download en_core_web_trf at the env

## Usage

- Train and save model data :

```python
from model7.rnn_package import train_rnn
from pathlib import Path

run_dir = Path("model7/saved_models")
run_dir.mkdir(parents=True, exist_ok=True)
run_dir  = "model7/saved_models"

data_dir = Path("model7/data")
data_dir.mkdir(parents=True, exist_ok=True)
data_dir = "model7/data"                 # data/train / data/test


res = train_rnn(data_dir, run_dir, n_epochs=3, batch_size=32)
print(res)
```

- Load a saved model and run predictions:

```python
texts = [
    'so add a fucking rick roll emte with the', 
'i rather the guy learn whatever game he enjoy the fastest. which be stupid lol',
'lucky colat i ve even get in',
'gta online update for june / july timeframe will cop and robber book it',
]

predictions = predict_texts(texts)  

print(predictions)  

#===============

df = load_and_preprocess_data("Twitter_data/testdata7.csv")
m7_pd = df.toPandas() 
testing_dataset_m7 = m7_pd["Phrase"].astype(str).tolist()   # <- now a plain list[str]



true_labels = m7_pd["Sentiment"].tolist()
print(true_labels)
#=========================================

predictions_7 = predict_texts(testing_dataset_m7)
```

## Notes

The package expects the dataset to be organized as `data/train/<label>/*.txt` and `data/test/<label>/*.txt`. 
The notebook `model7_data_setup_forrnn.ipynb` contains a helper to export CSV into this structure

Saved files include `fields.pth`, `label_field.pth`, `config.json`, and `best_acc.pt` / `best_loss.pt` in the provided `run_dir`.