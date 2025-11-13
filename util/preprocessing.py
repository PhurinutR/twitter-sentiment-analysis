from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, trim, udf
from pyspark.sql.types import *
import nltk
from nltk.stem import WordNetLemmatizer

# Goal: universally clean the tweets and output as a spark df

def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

def lemmatize(data_str):
    """
    Lemmatize text using NLTK WordNetLemmatizer with POS tagging.
    
    Args:
        data_str: Input string to lemmatize
    
    Returns:
        Lemmatized string
    """
    if not data_str or not isinstance(data_str, str):
        return ''
    
    try:
        lmtzr = WordNetLemmatizer()
        text = data_str.split()
        
        if not text:
            return ''
        
        tagged_words = nltk.pos_tag(text)
        lemmatized_words = []
        
        for word, tag in tagged_words:
            # Determine POS tag for better lemmatization
            if tag.startswith('V'):  # Verb
                lemma = lmtzr.lemmatize(word, pos='v')
            elif tag.startswith('J'):  # Adjective
                lemma = lmtzr.lemmatize(word, pos='a')
            elif tag.startswith('R'):  # Adverb
                lemma = lmtzr.lemmatize(word, pos='r')
            else:  # Default to noun
                lemma = lmtzr.lemmatize(word, pos='n')
            
            lemmatized_words.append(lemma)
        
        return ' '.join(lemmatized_words)
    except Exception as e:
        # If lemmatization fails, return original string
        print(f"Warning: Lemmatization failed: {e}")
        return data_str


def clean_tweets(df, text_column="Phrase"):
    """
    Clean Twitter text data for sentiment analysis.
    
    Args:
        df: Spark DataFrame containing tweet data
        text_column: Name of the column containing tweet text (default: "Phrase")
    
    Returns:
        Spark DataFrame with cleaned text
    """
    cleaned_df = df
    
    # Convert text column to string type if needed
    cleaned_df = cleaned_df.withColumn(text_column, col(text_column).cast(StringType()))
    
    # Remove URLs (http://, https://, www., t.co links)
    cleaned_df = cleaned_df.withColumn(
        text_column,
        regexp_replace(col(text_column), r'http\S+|www\.\S+|https?://\S+|t\.co/\S+', '')
    )
    
    # Remove user mentions (@username)
    cleaned_df = cleaned_df.withColumn(
        text_column,
        regexp_replace(col(text_column), r'@\w+', '')
    )
    
    # Remove HTML entities and special characters like <unk>
    cleaned_df = cleaned_df.withColumn(
        text_column,
        regexp_replace(col(text_column), r'<[^>]+>', '')
    )
    
    # Remove special unicode characters and normalize
    cleaned_df = cleaned_df.withColumn(
        text_column,
        regexp_replace(col(text_column), r'â\S+|â\S+|â\S+', '')
    )
    
    # Remove extra whitespace and newlines
    cleaned_df = cleaned_df.withColumn(
        text_column,
        regexp_replace(col(text_column), r'\s+', ' ')
    )
    
    # Convert to lowercase
    cleaned_df = cleaned_df.withColumn(
        text_column,
        lower(col(text_column))
    )
    
    # Trim leading/trailing whitespace
    cleaned_df = cleaned_df.withColumn(
        text_column,
        trim(col(text_column))
    )
    
    # Remove empty strings (after cleaning, some tweets might become empty)
    cleaned_df = cleaned_df.filter(col(text_column) != '')
    cleaned_df = cleaned_df.filter(col(text_column).isNotNull())

    # Lemmatize the text
    lem_word_udf = udf(lemmatize, StringType())
    cleaned_df = cleaned_df.withColumn(text_column, lem_word_udf(col(text_column)))
    return cleaned_df


def load_and_preprocess_data(file_path, text_column="Phrase", sentiment_column="Sentiment"):
    """
    Load CSV data and preprocess it for sentiment analysis.
    
    Args:
        file_path: Path to the CSV file
        text_column: Name of the text column (default: "Phrase")
        sentiment_column: Name of the sentiment column (default: "Sentiment")
    
    Returns:
        Spark DataFrame with cleaned data
    """

    # Download required NLTK data
    download_nltk_data()

    # Create Spark session
    ss = SparkSession.builder \
        .appName("TwitterSentimentAnalysis") \
        .getOrCreate()
    
    # Read CSV file
    df = ss.read.csv(
        file_path,
        header=True,
        inferSchema=True,
        quote='"',
        escape='"'
    )
    
    # Clean the tweets
    cleaned_df = clean_tweets(df, text_column)
    
    # Ensure sentiment column is numeric
    cleaned_df = cleaned_df.withColumn(
        sentiment_column,
        col(sentiment_column).cast(IntegerType())
    )

    return cleaned_df


# Example usage:
# train_df = load_and_preprocess_data("Twitter_data/traindata7.csv")
# test_df = load_and_preprocess_data("Twitter_data/testdata7.csv")

if __name__ == "__main__":
    train_df = load_and_preprocess_data("../Twitter_data/traindata7.csv")
    test_df = load_and_preprocess_data("../Twitter_data/testdata7.csv")
    train_df.show(10)
    test_df.show(10)