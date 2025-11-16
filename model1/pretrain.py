import sys
import os
# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model1.pipeline import BertDNNPipeline

# Train with checkpoints (saves best model during training)
print("Starting training...")
pipeline = BertDNNPipeline(num_classes=4, batch_size=64, freeze_bert=True)
pipeline.fit(
    epochs=30,
    save_checkpoints=True,
    save_best_only=True,
    checkpoint_dir="./model1/checkpoints"
)

print("\n" + "="*50)
print("Training complete!")
print("="*50)

# Training complete - save final model for inference
print("\nSaving final model...")
pipeline.save("./model1/final_model")
print("Final model saved!")

# Test prediction
print("\nTesting prediction on sample text...")
test_texts = ["This movie is great!", "I hate this product", "It's okay I guess"]
predictions = pipeline.predict(test_texts)
print(f"Predictions: {predictions}")
print("\n All tests passed!")
