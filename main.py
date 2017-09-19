"""
This is the main module of the project. It contains the entry point
CLI function of whole application.
"""


import sys
import numpy as np
import preprocess
import classify
import cluster
import draw
import settings
import pandas as pd
from geopy.distance import great_circle
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from scipy import interp


def _cluster(input_filename):
    return cluster.by_grid(input_filename, settings.CLUSTER.OUTPUT_FILENAME,
                           map_bounds=settings.MAP.BOUNDS, by_depth=settings.CLUSTER.BY_DEPTH)


def _learn(input_filename, centroids, is_hybrid):
    # prepare the data
    corpus, labels = preprocess.read_corpus(input_filename)
    embedder = classify.TextEmbedder(is_hybrid)
    corpus, labels = embedder(corpus, labels)

    # split data and learn the model
    X, y = corpus, labels
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=settings.CLASSIFY.TEST_SIZE,
                                                        random_state=settings.CLASSIFY.RANDOM_SEED)

    tfidf = preprocess.Vectorizer()
    logreg = classify.learn(tfidf.fit(X_train), y_train)

    # predict labels
    X_transformed = tfidf.transform(X)
    X_test_transformed = tfidf.transform(X_test)
    y_pred = logreg.predict(X_transformed)
    y_test_pred = logreg.predict(X_test_transformed)
    accuracy, _ = classify.test(logreg, X_test_transformed, y_test)
    kappa = cohen_kappa_score(y_test, y_test_pred)

    _roc(X_transformed, y)

    # save predicted labels
    original_df = preprocess.read_dataframe(input_filename)
    text = original_df[settings.CSV.INPUT.TEXT]
    actual_latitude = original_df[settings.CSV.OUTPUT.ACTUAL_LATITUDE]
    actual_longitude = original_df[settings.CSV.OUTPUT.ACTUAL_LONGITUDE]
    y_pred = pd.Series(y_pred, name=settings.CSV.OUTPUT.PREDICTED_GRID)
    predicted_latitude = pd.Series(
        [centroids[grid][settings.CSV.INPUT.LATITUDE] for grid in y_pred],
        name=settings.CSV.OUTPUT.PREDICTED_LATITUDE
    )
    predicted_longitude = pd.Series(
        [centroids[grid][settings.CSV.INPUT.LONGITUDE] for grid in y_pred],
        name=settings.CSV.OUTPUT.PREDICTED_LONGITUDE
    )

    error = pd.Series(
        list(map(lambda x, y: round(great_circle(x, y).kilometers, 2),
                 zip(actual_latitude, actual_longitude), zip(
                     predicted_latitude, predicted_longitude)
                 )),
        name=settings.CSV.OUTPUT.ERROR
    )

    dataframe = pd.concat([text,
                           actual_latitude,
                           actual_longitude,
                           predicted_latitude,
                           predicted_longitude,
                           error,
                           labels,
                           y_pred], axis=1)

    output_filename = settings.CLASSIFY.OUTPUT_FILENAME
    dataframe.to_csv(output_filename, sep='\t', header=True)
    return accuracy, kappa, (y_test, y_test_pred)


def _visualize(y_values, tree, map_bounds, output_filename):
    draw.draw_result(y_values, tree, max_depth=tree.height,
                     map_bounds=map_bounds)
    plt.savefig(output_filename, format='svg', dpi=2000)
    print("Map file:", output_filename)


def _roc(X, y):
    # Binarize the output
    y = label_binarize(y, classes=sorted(y.unique()))
    n_classes = y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(LogisticRegression())
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    output_filename = settings.CLASSIFY.ROC_FILENAME
    plt.savefig(output_filename, format='svg', dpi=2000)
    print("ROC file:", output_filename)


def cli():
    """
    The entry point function.
    """
    input_filename = sys.argv[1]

    # cluster corpus by geocoordinates
    tree, centroids = _cluster(input_filename)

    # learn model and predict geolabels by text content
    accuracy, kappa, y_values = _learn(settings.CLUSTER.OUTPUT_FILENAME,
                                       centroids,
                                       settings.CLASSIFY.USE_HYBRID)
    print("k-fold CV accuracy:", accuracy)
    print("Cohen Kappa score:", kappa)

    # draw a map
    _visualize(y_values, tree,
               settings.MAP.BOUNDS, settings.MAP.OUTPUT_FILENAME)


if __name__ == "__main__":
    cli()
