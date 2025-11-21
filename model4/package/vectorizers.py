from sklearn.feature_extraction.text import CountVectorizer

def get_count_vectorizer(ngram=(1,1), min_df=1, max_features=None):
    return CountVectorizer(
        ngram_range=ngram,
        min_df=min_df,
        max_features=max_features,
        stop_words="english"
    )
