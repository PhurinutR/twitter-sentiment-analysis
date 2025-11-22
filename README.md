# Twitter-sentiment-analysis

## Getting Started
After you cloned this GitHub repository, please run the following command to start playing with this project.

```bash
cd twitter-sentiment-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then, install PyTorch that's appropriate for your machine. If you are on Linux you can run the following command:
(Note: This command is for CUDA 13.0)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Large Model Weight Installation Guides
For model1 which is BERT + DNN (ResNet), please see this [model 1 installation guideline](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model1).

For model3 which is Word2Vec + Random Forest, please see this [model 3 installation guideline](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model3).

### Guidelines on How to Use the Models

1. BERT + DNN (ResNet): [Model 1 installation guideline](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model1).

2. GloVe + LSTM: [Guide in model 2](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model2).

3. Word2Vec + Random Forest: [Guide in model 3](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model3).

4. CountVectorizer + Decision Tree: [Guide in model 4](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model4).

5. TF-IDF + Naive Bayes: [Guide in model 5](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model5).

6. DFIDF + Logistic Regression and Word2Vec + SVM (Baseline): [Guide in model 6](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model6).

7. GloVe + RNN: [Guide in model 7](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model7).