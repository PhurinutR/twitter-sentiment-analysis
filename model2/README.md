# Model2 LSTM Implementation

This folder provides a Python package that encapsulates the GloVe embedding, LSTM training, and inference workflow.

The main package file: `lstm_package/model.py` contains the core architecture of the implemetation, the details as follows:
  - `train_lstm(data_dir, run_dir, **kwargs)`: trains the model with configurable hyperparameters via `**kwargs`, saves model data and Fields, and returns accuracy and loss metrics.
  - `predict_texts(text_list)`: loads saved best model and reconstructs the model ready for, classify list of texts data into sentiments, then obtain the predictions.


## Prerequisite

1. Make sure to install all necessary libraries by following the **"Getting Started"** guide in the repo home page.
2. Pre-process the training data first
3. run python -m spacy download en_core_web_sm and python -m spacy download en_core_web_trf at the env
## Usage

- Train and save model data to `run_dir`:

```python
from model2.lstm_package import train_lstm
result = train_lstm('data', 'model2/data', n_epochs=5, batch_size=32)
print(result)
```

- Load a best model(best: embedding_dim: 2700,2_layers, bidirectional, dropout: 0.5) and run predictions:

```python
from model2.lstm_package import predict_texts
texts = [

'i rather the guy learn whatever game he enjoy the fastest. which be stupid lol',
'lucky colat i ve even get in',
'gta online update for june / july timeframe will cop and robber book it',
]

predictions = predict_texts(texts)  


print(predictions)  
#=======================
df = load_and_preprocess_data("Twitter_data/testdata7.csv")
m2_pd = df.toPandas() 
testing_dataset_m2 = m2_pd["Phrase"].astype(str).tolist()   # <- now a plain list[str]
# print(testing_dataset)

# predictions : list of model outputs 
predictions_2 = predict_texts(testing_dataset_m2)
print(predictions_2)        # raw probabilities (0-1)

# true_labels : list of gold labels
true_labels = m2_pd["Sentiment"].tolist()
print(true_labels)

```

## Notes

The package expects the dataset to be organized as `data/train/<label>/*.txt` and `data/test/<label>/*.txt`. 
The notebook `model2_data_setup_forlstm.ipynb` contains a helper to export CSV into this structure — use it if your data is in CSV form.

Saved files include `fields.pth`, `label_field.pth`, `config.json`, and `best_acc.pt` / `best_loss.pt` in the provided `run_dir`.