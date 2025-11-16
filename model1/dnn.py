import torch
import torch.nn as nn
from transformers import BertModel
import os
from typing import List
import json

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Use a projection layer if dimensions don't match
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = None
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.linear(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Skip connection
        if self.projection is not None:
            identity = self.projection(identity)
        
        # Add skip connection
        out = out + identity
        out = self.relu(out)  # ReLU after addition
        
        return out

class DNNHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Build residual blocks
        blocks = []
        prev_dim = input_dim
        for h in hidden_dims:
            blocks.append(ResidualBlock(prev_dim, h, dropout))
            prev_dim = h
        
        self.blocks = nn.Sequential(*blocks)
        
        # Final classification layer (no skip connection here)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        logits = self.classifier(x)
        
        if self.training:
            return logits  # Returns logits
        else:
            return torch.softmax(logits, dim=-1)  # Returns probabilities

class BertDNN(nn.Module):
    def __init__(self, bert: BertModel, head: nn.Module, freeze_bert: bool = False):
        super().__init__()
        self.bert = bert
        self.head = head
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, tokenized_phrase):
        outputs = self.bert(**tokenized_phrase)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.head(pooled)
        return logits