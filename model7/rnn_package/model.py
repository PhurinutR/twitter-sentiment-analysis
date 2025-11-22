"""
LSTM training, persistence, and inference helpers.

Modules:
- `train_lstm` to train and save model data (state_dict, Fields, config).
- `load_lstm` to restore model and Fields for inference.
- `predict_sentiment` to classify raw text strings using a loaded model.

Notes:
- This code expects `torch` and `torchtext` to be installed via requirements.txt. 
"""

import os
import json
from typing import List
import time
import torch
import torch.nn as nn
from pathlib import Path
# Use torchtext.legacy.data if it exists, otherwise torchtext.data
try:
    # torchtext >= 0.9 exposes legacy.data
    from torchtext.legacy import data
except Exception:
    # torchtext < 0.9 (0.5.0 in this case)
    from torchtext import data

class FolderDataset(data.Dataset):
    """
    Dataset that reads one text file per example from folders named by label.
    Expects this directory structure within model2:
        data/
        ├── train/
        │     ├── classA/
        │     │     ├── xxx.txt
        │     │     └── yyy.txt
        │     └── classB/
        └── test/
                └── ...
    """

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, root, text_field, label_field, encoding="utf-8", **kwargs):
        fields = [("text", text_field), ("label", label_field)]
        examples = []

        root = os.path.abspath(root)
        for class_dir in sorted([p for p in os.scandir(root) if p.is_dir()], key=lambda p: p.name):
            label = class_dir.name
            for entry in sorted(os.scandir(class_dir.path), key=lambda e: e.name):
                if entry.is_file() and entry.name.endswith('.txt'):
                    txt = open(entry.path, encoding=encoding).read()
                    examples.append(data.Example.fromlist([txt, label], fields))

        super(FolderDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, path, train="train", test="test", **kwargs):
        train_ds = cls(os.path.join(path, train), text_field, label_field, **kwargs)
        test_ds = cls(os.path.join(path, test), text_field, label_field, **kwargs)
        return train_ds, test_ds



def train_rnn(
    data_dir: str,
    run_dir: str,
    embedding_dim: int = 300,
    hidden_dim: int = 256,
    n_layers: int = 2,
    bidirectional: bool = True,
    dropout: float = 0.5,
    batch_size: int = 64,
    n_epochs: int = 125,
    max_vocab_size: int = 250_000,
    pretrained_vectors: str = "glove.840B.300d"
):
    """
    Train an LSTM model and save the model locally.

    Args:
        data_dir: a folder containing `train` and `test` folders organized by labels. Required.
        run_dir: output directory where model, fields, and config will be saved. Required.
        Other args: hyperparameters for model training. Optional, defaults provided.

    Returns:
        A dictionary with best metrics and saved paths: {'best_test_acc': ..., 'best_test_loss': ..., 'data': {...}}
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(run_dir, exist_ok=True)

    TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
    LABEL = data.LabelField(dtype=torch.long)

    print('Loading datasets...')
    train_data, test_data = FolderDataset.splits(TEXT, LABEL, path=data_dir)

    # Build vocab with pre-trained embeddings - GloVe
    print('Building vocabulary...')
    TEXT.build_vocab(train_data, max_size=max_vocab_size, vectors=pretrained_vectors, unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)

    INPUT_DIM = len(TEXT.vocab)
    OUTPUT_DIM = len(LABEL.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    train_iter, test_iter = data.BucketIterator.splits((train_data, test_data), batch_size=batch_size, sort_within_batch=True, device=device)

    model = RNNClassifier(INPUT_DIM, embedding_dim, hidden_dim, OUTPUT_DIM, n_layers, bidirectional, dropout, pad_idx=PAD_IDX)
    model = model.to(device)

    # Initialize embeddings for UNK and PAD to zeros if pre-trained is used
    try:
        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)
    except Exception:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    best_test_acc = float('-inf')
    best_test_loss = float('inf')
    best_acc_path = os.path.join(run_dir, 'best_acc.pt')
    best_loss_path = os.path.join(run_dir, 'best_loss.pt')

    print('Starting training...')
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for batch in train_iter:
            optimizer.zero_grad()
            text, text_lengths = batch.text
            predictions = model(text, text_lengths)
            loss = criterion(predictions, batch.label)
            preds = predictions.argmax(dim=1)
            acc = (preds == batch.label).float().mean().item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc
        train_loss = epoch_loss / len(train_iter)
        train_acc = epoch_acc / len(train_iter)

        # evaluation
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for batch in test_iter:
                text, text_lengths = batch.text
                output = model(text, text_lengths)
                loss = criterion(output, batch.label)
                preds = output.argmax(dim=1)
                acc = (preds == batch.label).float().mean().item()
                test_loss += loss.item()
                test_acc += acc
        test_loss = test_loss / len(test_iter)
        test_acc = test_acc / len(test_iter)

        print(f"Epoch {epoch} | Train loss={train_loss:.4f} acc={train_acc:.4f} | Test loss={test_loss:.4f} acc={test_acc:.4f}")

        # Save best results
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_acc_path)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), best_loss_path)

    # After training, save result: lstm model fields and config
    result = {}
    fields_path = os.path.join(run_dir, 'fields.pth')
    torch.save(TEXT, fields_path)
    torch.save(LABEL, os.path.join(run_dir, 'label_field.pth'))
    result['fields'] = fields_path
    result['label_field'] = os.path.join(run_dir, 'label_field.pth')
    result['best_acc'] = best_acc_path
    result['best_loss'] = best_loss_path

    config = {
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'bidirectional': bidirectional,
        'dropout': dropout,
        'pad_idx': PAD_IDX,
        'vocab_size': INPUT_DIM,
        'output_dim': OUTPUT_DIM,
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    return {'best_test_acc': best_test_acc, 'best_test_loss': best_test_loss, 'data': result, 'config': config}

# model2/lstm_package/model_inference.py



# Define the LSTM-based model (same architecture as used in training)
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        # Embedding layer with padding index
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # Recurrent layer: stacked RNN
        self.rnn = nn.RNN(embedding_dim, hidden_dim, 
                          num_layers=n_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout if n_layers > 1 else 0.0)
        # Linear layer input size depends on bidirectional
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

    def forward(self, text, text_lengths):
        # text: [sentence_length, batch_size]
        # text_lengths: [batch_size]
        embedded = self.dropout(self.embedding(text))
        # Pack the sequence of embeddings (for variable-length sequence support)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        packed_output, hidden = self.rnn(packed_embedded)
        # Unpack the sequence (we won't use the padded output, so just call pad_packed_sequence without assignment)
        nn.utils.rnn.pad_packed_sequence(packed_output)
        # hidden shape: [n_layers * num_directions, batch_size, hidden_dim]
        if self.bidirectional:
            # Concatenate final forward and backward hidden states from the last layer
            hidden_combined = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # Use the final hidden state from the last layer
            hidden_combined = hidden[-1, :, :]
        # Apply dropout to the final hidden state and feed through the fully connected layer
        hidden_dropped = self.dropout(hidden_combined)
        return self.fc(hidden_dropped)



def predict_texts(text_list):
    """
    Predict sentiment labels for each text in text_list.
    Returns a list of predictions (as integers 0, 1, 2, 3, etc.) in the same order as the input.
    """

    # Device configuration (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved TorchText Field objects for text and labels
    # ----------------------------------------------------------
    PKG_DIR   = Path(__file__).resolve().parent       # .../lstm_package
    ROOT_DIR  = PKG_DIR.parent                        # project root
    SAVED_DIR = ROOT_DIR / "saved_best"

    # Sanity-check: raise a helpful error if files are missing
    required = ["fields.pth", "label_field.pth", "best_acc.pt"]
    missing  = [f for f in required if not (SAVED_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Could not find {', '.join(missing)} in {SAVED_DIR}. "
            "Make sure the saved weights are located there."
        )

    # ----------------------------------------------------------
    # 3. Load TorchText Field objects
    # ----------------------------------------------------------
    TEXT  = torch.load(SAVED_DIR / "fields.pth",       map_location=device, weights_only = False)
    LABEL = torch.load(SAVED_DIR / "label_field.pth",  map_location=device, weights_only = False)

    # Reconstruct the model and load trained weights
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 900   # must match the embedding dim used during training
    HIDDEN_DIM = 256       # must match hidden dim from training
    OUTPUT_DIM = len(LABEL.vocab)   # number of classes (e.g., 4 if labels 0-3)
    N_LAYERS = 2           # number of LSTM layers as used in training
    BIDIRECTIONAL = False   # should match training (True for bi-LSTM)
    DROPOUT = 0.5          # dropout probability used in training
    PAD_IDX = TEXT.vocab.stoi.get(TEXT.pad_token, 1)  # index of <pad> token (default to 1 if not found)

    # Initialize model and load parameters
    model = RNNClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, 
                        OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, pad_idx=PAD_IDX)
    model.load_state_dict(torch.load(SAVED_DIR /"best_acc.pt", map_location=device))
    model.to(device)
    model.eval()  # set model to evaluation mode
    predictions = []
    for text in text_list:
        # Tokenize the text using the same tokenizer as used in training
        tokens = [tok.lower() for tok in TEXT.tokenize(text)]
        # Convert tokens to indices using the vocabulary (use <unk> index for unseen words)
        indices = [TEXT.vocab.stoi.get(tok, TEXT.vocab.stoi[TEXT.unk_token]) for tok in tokens]
        # Create tensors for model input
        text_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)   # shape: [seq_len, batch_size=1]
        text_length = torch.LongTensor([len(indices)]).to(device)         # shape: [batch_size=1]
        # Get model prediction
        with torch.no_grad():
            logits = model(text_tensor, text_length)        # raw logits for each class
            pred_idx = int(logits.argmax(dim=1).item())     # predicted class index (as int)
        # If label vocabulary is numeric (e.g., "0", "1", ...), convert to int; otherwise use as is
        pred_label_token = LABEL.vocab.itos[pred_idx]
        try:
            pred_label = int(pred_label_token)
        except ValueError:
            pred_label = pred_label_token
        predictions.append(pred_label)
    return predictions
