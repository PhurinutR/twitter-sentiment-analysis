import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from typing import List
import json
from pathlib import Path
import sys
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, mode="train", tokenizer=None, max_length: int=256):

        self.train_path = "traindata7.csv"
        self.test_path = "testdata7.csv"
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # add the project root to the path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        from util.preprocessing import load_and_preprocess_data

        # revert the path to the original
        sys.path.pop(0)

        if mode == "train":
            self.data_df = load_and_preprocess_data(str(project_root / "Twitter_data" / self.train_path))
        elif mode == "test":
            self.data_df = load_and_preprocess_data(str(project_root / "Twitter_data" / self.test_path))

        self.data_df = self.data_df.collect()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        phrase = self.data_df[idx]["Phrase"]
        sentiment = self.data_df[idx]["Sentiment"]

        tokenized_phrase = self.tokenizer(
            phrase,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Squeeze to remove batch dimension added by return_tensors='pt'
        return {
            "input_ids": tokenized_phrase['input_ids'].squeeze(0),
            "attention_mask": tokenized_phrase['attention_mask'].squeeze(0),
            "labels": torch.tensor(sentiment, dtype=torch.long)
        }

if __name__ == "__main__":
    '''
    Test the TextDataset class
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextDataset(mode="train", tokenizer=tokenizer)
    print(len(dataset))
    print(dataset[0])