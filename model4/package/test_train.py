# Train Model 4 (Decision Tree + CountVectorizer) and save the final pipeline.

import os
import joblib
from model.package import load_dataset, clean_text, DecisionTreeSentiment

# ---------------------------------------------------------
# Paths (adjust if needed)
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)
# ---------------------------------------------------------

# 1. Load dataset
X_train, y_train, X_test, y_test = load_dataset(TRAIN_PATH, TEST_PATH)

# 2. Clean text
X_train = X_train.apply(clean_text)
X_test = X_test.apply(clean_text)

# 3. Train Decision Tree Model
dt_model = DecisionTreeSentiment()
best_params, best_cv_score = dt_model.train(X_train, y_train)

print("Best Hyperparameters:", best_params)
print("Best CV Score:", best_cv_score)

# 4. Evaluate
test_acc, test_err = dt_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Error Rate: {test_err:.4f}")

# 5. Save the final full pipeline (vectorizer + classifier)
save_path = os.path.join(SAVE_DIR, "decision_tree_pipeline.joblib")
joblib.dump(dt_model.model, save_path)
print(f"Model saved to: {save_path}")
