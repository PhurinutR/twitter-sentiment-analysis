"""
Tester for `model2.lstm_package`.

Run from repo root directory:
python -m model2.lstm_package.example_usage
"""

import os, sys, pprint

# Import packages. If import fails, add root path to sys.path
try:
    from model2.lstm_package import train_lstm, load_lstm, predict_sentiment
except Exception:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    from model2.lstm_package import train_lstm, load_lstm, predict_sentiment

def main():
    data_dir = "data"                   # expects data/train/<label>/*.txt and data/test/<label>/*.txt
    run_dir = "model2/saved_models"     # directory to save model and Fields

    # Train LSTM model
    # play around with these parameters! Especially the epochs
    res = train_lstm(data_dir, run_dir, embedding_dim=300, hidden_dim=128, n_epochs=3, batch_size=32)
    pprint.pprint(res)

    # Load a trained LSTM
    data = load_lstm(run_dir)
    model = data['model']
    TEXT = data['TEXT']
    print("Loaded model:", model, "with TEXT field:", TEXT, "config:", data['config'])

    # Predict some sample texts
    sample_texts = ["I love this product", "This is awful"]
    preds = predict_sentiment(model, TEXT, sample_texts)
    print("Predictions:", preds)

if __name__ == '__main__':
    main()
