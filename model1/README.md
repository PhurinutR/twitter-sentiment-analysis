# How to Use Model 1: *BERT + DNN (ResNet)*
Author: *Phurinut Rungrojkitiyos* <br/>
Core Architecture: *BERT + DNN (with residual connection like ResNet)*

## ⚠️ Pre-requisite
Before, running anything here please install pytorch and run the following command to install other necessary packages first.

```bash
pip install -r requirements.txt
```

Then install the weighting using the following command:

```bash
cd twitter-sentiment-analysis
```

```bash
hf download PhurinutR/twitter-sentiment-analysis \
  --include "checkpoints/**" \
  --local-dir ./model1
```

```bash
hf download PhurinutR/twitter-sentiment-analysis \
  --include "final_model/**" \
  --local-dir ./model1
```

```bash
hf download PhurinutR/twitter-sentiment-analysis \
  --include "final_model_finetuned/**" \
  --local-dir ./model1
```

## How to use BertDNNPipeline

### Here is how to initialize the pipeline and load the final fine-tuned model:

```python
from model1.pipeline import BertDNNPipeline
pipeline = BertDNNPipeline.load(
    "./model1/final_model_finetuned",
    head_hidden_dims=[512, 512, 256, 128, 64, 32, 16],
    num_classes=4
)
```

### To run a prediction, you can do like this:
```python
test_texts = ["This movie is great!", "I hate this product", "It's okay I guess"]
predictions = pipeline.predict(test_texts)
```

### To evaluate the model accuracy using test dataset, you can do like this:
```python
from model1.text_dataset import TextDataset
from torch.utils.data import DataLoader
test_dataset = TextDataset(mode="test", tokenizer=pipeline.tokenizer, max_length=pipeline.max_length)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
test_acc = pipeline.evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.4f}")
```

### To fine-tune the model further please see the example below:
```python
epoch, loss, val_acc = pipeline.load_checkpoint("./model1/checkpoints/best_model")
val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
print(f"Resuming from epoch {epoch} with val_acc: {val_acc_str}")

# Continue training
pipeline.fit(
    epochs=20,
    save_checkpoints=True,
    save_best_only=True,
    checkpoint_dir="./model1/checkpoints",
    use_scheduler=True,
    scheduler_type="cosine"
)

```