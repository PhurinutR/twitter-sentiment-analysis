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
1. For model1 which is BERT + DNN (ResNet), please see this [model 1 installation guideline](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model1).

2. For model2 which is GloVe and (Bidirectional) LSTM, please see the [guide in model 2](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model2).

3. For model3 which is Doc2Vec + Random Forest, please see the [guide in model 3](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model3).

4. For model4 which is CountVectorizer + Decision Tree, please refer to the [implementation guide in model 4](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model4).

5. For model5 which is TF-IDF + Naive Bayes, please follow the [implementation guide in model 5](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model5).

6. For model6 which is DFIDF + Logistic Regression / Word2Vec + SVM / DFIDF + SVM, refer to the [guide in model 6](https://github.com/PhurinutR/twitter-sentiment-analysis/tree/main/model6).