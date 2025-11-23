# Model7 RNN Implementation

This folder provides a Python package that encapsulates the GloVe embedding, RNN training, and inference workflow.

The main package file: `rnn_package/model.py` contains the core architecture of the implemetation, the details as follows:
  - `train_rnn(data_dir, run_dir, **kwargs)`: trains the model with configurable hyperparameters via `**kwargs`, saves model data and Fields, and returns accuracy and loss metrics.
  - `predict_texts(text_list)`: loads saved best model and reconstructs the model ready for, classify list of texts data into sentiments, then obtain the predictions.


## Prerequisite

1. Make sure to install all necessary libraries by following the **"Getting Started"** guide in the repo home page.
2. Pre-process the training data first, by running

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

# train
"""
Please run "model2_data_setup_LSTM.ipynb" from model2 before executing this cell!
"""
res = train_rnn(data_dir, run_dir, n_epochs=3, batch_size=32)
print(res)
```

- Load a saved model and run predictions:

```python
from model2.lstm_package import load_lstm, predict_sentiment
data = load_lstm('model2/data')
model = data['model']
TEXT = data['TEXT']
preds = predict_sentiment(model, TEXT, ['I love this', 'This is bad', 'Sounds alright.'])
print(preds)
```

## Notes

The package expects the dataset to be organized as `data/train/<label>/*.txt` and `data/test/<label>/*.txt`. 
The notebook `model2_data_setup_forlstm.ipynb` contains a helper to export CSV into this structure — use it if your data is in CSV form.

Saved files include `fields.pth`, `label_field.pth`, `config.json`, and `best_acc.pt` / `best_loss.pt` in the provided `run_dir`.