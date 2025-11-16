# Continue training the model from the last checkpoint
from pipeline import BertDNNPipeline
from text_dataset import TextDataset
from torch.utils.data import DataLoader

# 1. Create a new pipeline with the same configuration
pipeline = BertDNNPipeline(
    num_classes=4,
    batch_size=8,
    lr=2e-4,
    head_hidden_dims=[512, 512, 256, 128, 64, 32, 16],
    wandb_project="bert-dnn-pipeline",
    wandb_run_name="continue-training"
)

# 2. Load checkpoint to resume training
epoch, loss, val_acc = pipeline.load_checkpoint("./model1/checkpoints/best_model")
val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
print(f"Resuming from epoch {epoch} with val_acc: {val_acc_str}")

# 3. Continue training
pipeline.fit(
    epochs=30,
    save_checkpoints=True,
    save_best_only=True,
    checkpoint_dir="./model1/checkpoints",
    use_scheduler=True,
    scheduler_type="cosine"
)

# 4. Save the final fine-tuned model
pipeline.save("./model1/final_model_finetuned")
print("✓ Final model saved!")

# 5. Load the final fine-tuned model and evaluate it
print("\nEvaluating on test set...")
pipeline_eval = BertDNNPipeline.load(
    "./model1/final_model_finetuned",
    head_hidden_dims=[512, 512, 256, 128, 64, 32, 16],
    num_classes=4
)
test_dataset = TextDataset(mode="test", tokenizer=pipeline_eval.tokenizer, max_length=pipeline_eval.max_length)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
test_acc = pipeline_eval.evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.4f}")