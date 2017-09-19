"""
I/O functions
"""

import re
import pandas as pd
import nltk
import settings
import joblib
import csv
from sklearn.feature_extraction.text import TfidfVectorizer, \
    TfidfTransformer, \
    CountVectorizer


def get_stop_words():
    """
    Setup nltk on import
    """
    nltk.download('stopwords')
    return set(nltk.corpus.stopwords.words('english'))


STOP_WORDS = get_stop_words()


def read_dataframe(input_filename):
    """
    Read a csv file as pandas dataframe.
    """
    with open(input_filename, 'r'):
        return pd.read_csv(input_filename, sep='\t', quoting=csv.QUOTE_NONE)


def _tokenize(tweet):
    """
    Split a tweet text into tokens, and remove stop words
    """
    tweet = re.sub(r"http\S+", "", tweet)
    tokens = re.findall(r'\w+', tweet.lower())
    return [token for token in tokens if not token in STOP_WORDS]


def read_corpus(input_filename):
    """
    Read and preprocess tweet dataset.
    """
    dataset = read_dataframe(input_filename)
    dataset[settings.CSV.OUTPUT.TEXT] = dataset[settings.CSV.OUTPUT.TEXT].apply(
        _tokenize)
    return dataset[settings.CSV.OUTPUT.TEXT], dataset[settings.CSV.OUTPUT.ACTUAL_GRID]


class Vectorizer:
    """
    Vectorize corpus by transforming into TF-IDF matrix
    """

    def __init__(self):
        self.vocabulary = None

    def _load_vocabulary(self):
        self.vocabulary = joblib.load(settings.CLASSIFY.TFIDF_FILENAME)

    def _join_words(self, corpus):
        return [' '.join(tokens for tokens in tweet) for tweet in corpus]
    
    def fit(self, corpus):
        """
        Fit and transform new data to TF-IDF matrix
        """
        vectorizer = CountVectorizer(decode_error="replace")
        corpus = self._join_words(corpus)
        vec_train = vectorizer.fit_transform(corpus)
        
        # save vocabulary
        self.vocabulary = vectorizer.vocabulary_
        joblib.dump(vectorizer.vocabulary_, settings.CLASSIFY.TFIDF_FILENAME)

        return vec_train

    def transform(self, corpus):
        """
        Transform to TF-IDF matrix using only word vocabulary of previous fit
        """
        transformer = TfidfTransformer()
        if not self.vocabulary:
            self._load_vocabulary()

        loaded_vec = CountVectorizer(decode_error="replace",
                                     vocabulary=self.vocabulary)

        corpus = self._join_words(corpus)
        return transformer.fit_transform(loaded_vec.fit_transform(corpus))
