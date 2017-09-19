"""
This module is aimed to predict tweets label based on its content.
"""


from __future__ import print_function

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from gensim.models import word2vec
import settings


def _jaccard(x, y):
    """
    Generalized Jaccard index of similarity
    """
    return sum(min(x[i], y[i]) for i in range(len(x))) / sum(max(x[i], y[i]) for i in range(len(x)))


def _normalize(model):
    """
    Normalize word embeddings
    """
    result = {}
    for word in model.wv.vocab:
        result[word] = model.wv[word] + abs(min(model.wv[word]))
    return result


class TextEmbedder:
    """
    Word2Vec embedding for tweet words, and reducing the dictionary
    """

    def __init__(self, is_hybrid):
        self.is_hybrid = is_hybrid
        self.model = None
        self.normalized_wv = None

    def __call__(self, corpus, labels):
        """
        Reduce total count of words used in corpus by testing their
        word2vec embeddings on high similarity
        """
        self.model = self._embed(corpus)
        self.normalized_wv = _normalize(self.model)
        corpus = self._dimreduce(corpus)
        return corpus, labels

    def _embed(self, corpus):
        """
        Create word2vec embeddings for words from the corpus
        """

        model = word2vec.Word2Vec(corpus,
                                  min_count=settings.CLASSIFY.W2V_MIN_COUNT,
                                  size=settings.CLASSIFY.W2V_SIZE,
                                  window=settings.CLASSIFY.W2V_WINDOW,
                                  workers=settings.CLASSIFY.W2V_WORKERS)
        model.save(settings.CLASSIFY.W2V_FILENAME)
        return model

    def _replace_condition(self, word, synonim):
        """
        Create a boolean condition of word replacement
        """
        cosin_sim = self.model.similarity(word, synonim)
        jaccard_sim = _jaccard(self.normalized_wv[word],
                               self.normalized_wv[synonim])

        condition = cosin_sim >= settings.CLASSIFY.COSINE_THRESHOLD and \
            self.model.wv.vocab[synonim].count > self.model.wv.vocab[word].count
        if self.is_hybrid:
            condition = condition and \
                jaccard_sim >= settings.CLASSIFY.JACCARD_THRESHOLD

        return condition

    def _dimreduce(self, corpus):
        """
        Replace similar words by their synonyms
        """

        to_replace = {}
        for word in self.model.wv.vocab:
            synonim, _ = self.model.wv.most_similar(word)[0]
            if self._replace_condition(word, synonim):
                to_replace[word] = synonim

        for i in range(len(corpus)):
            corpus[i] = [to_replace[word]
                         if word in to_replace else word for word in corpus[i]]

        # save fixed words
        with open("dictionary.txt", 'w') as g:
            print("\n".join("{}\t{}".format(k, v)
                            for k, v in to_replace.items()), file=g)

        print("Words eliminating:", len(self.model.wv.vocab), "->",
              len(self.model.wv.vocab) - len(to_replace))
        return corpus


def learn(X, y):
    """
    Learn logit model to predict labels by vectorized corpus
    """

    with open('ylog.txt', 'w') as ylog:
        print(y, file=ylog)

    with open('Xlog.txt', 'w') as xlog:
        print(X, file=xlog)

    # Fit Logistic Regression model to the dataset
    logreg = LogisticRegression()
    logreg.fit(X, y)

    # save the logit model
    joblib.dump(logreg, settings.CLASSIFY.LOGIT_FILENAME)
    return logreg


def test(logreg, X, y):
    """
    Test accuracy of prediction
    """
    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(
        estimator=logreg,
        X=X, y=y,
        cv=settings.CLASSIFY.CROSS_VALID_K)

    with open(settings.CLASSIFY.CV_ACCURACY_FILENAME, 'w') as output:
        print(accuracies.mean(), file=output)

    # calculate accuracy of class predictions
    pre_rec_fm = metrics.classification_report(y, logreg.predict(X))
    with open(settings.CLASSIFY.F1_FILENAME, 'w') as output:
        print(pre_rec_fm, file=output)

    return accuracies.mean(), pre_rec_fm


if __name__ == "__main__":
    raise RuntimeError("This module is not supposed to be called.")
