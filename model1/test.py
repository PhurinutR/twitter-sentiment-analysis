import sys
import os
# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model1.pipeline import BertDNNPipeline
from model1.text_dataset import TextDataset
from torch.utils.data import DataLoader

pipeline_eval = BertDNNPipeline.load(
    "./model1/final_model",
    head_hidden_dims=[512, 512, 256, 128, 64, 32, 16],
    num_classes=4
)
test_dataset = TextDataset(mode="test", tokenizer=pipeline_eval.tokenizer, max_length=pipeline_eval.max_length)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
test_acc = pipeline_eval.evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.4f}")