# How to Use Model 6: *Logistic Regression Model & SVM Model*

Author: *Komine Shunji*

Core Architecture: *DFIDF + Logistic Regression* / *Word2Vec + SVM* / *DFIDF + SVM*

## How to use the models

### DFIDF + Logistic Regression (Baseline)

```python
from pyspark.ml import PipelineModel

# Load the model
pipeline_model = PipelineModel.load("model6/baseline")

# Example text string to convert to DataFrame
input_text= [("I love this amazing thing!", "I hate whatever that was")]
new_df = spark.createDataFrame(input_text, ["Phrase"])

# Preprocess data
from util.preprocessing import clean_tweets
cleaned_new_df = clean_tweets(new_df, text_column="Phrase")

# Predict
predictions = pipeline_model.transform(cleaned_new_df)
predictions.select("Phrase", "prediction").show()
```

### DFIDF + SVM

```python
# Change above example with this line to use 'dfidf_svm'
# Load the model
pipeline_model = PipelineModel.load("model6/dfidf_svm")
```

### Word2Vec + SVM
```python
# Change above example with this line to use 'word2vec_svm'
# Load the model
pipeline_model = PipelineModel.load("model6/word2vec_svm")
```

### Evaluating the model accuracy
```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from util.preprocessing import load_and_preprocess_data

test_df = load_and_preprocess_data("/Twitter_data/testdata7.csv")
predictions = pipeline_model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(
    labelCol="Sentiment", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")
```
