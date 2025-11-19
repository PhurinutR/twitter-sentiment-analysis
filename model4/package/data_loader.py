import pandas as pd

def load_dataset(train_path, test_path, text_col="text", label_col="label"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[text_col]
    y_train = train_df[label_col]

    X_test = test_df[text_col]
    y_test = test_df[label_col]

    return X_train, y_train, X_test, y_test