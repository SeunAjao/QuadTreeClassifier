"""
Settings and constant values of the project.
"""

from bunch import Bunch

# Geographic map parameters
MAP = Bunch({
    
    # Including Alaska
    #'BOUNDS': ((5.49955, 83.162102), (-167.27641, -52.23304)),

    # Only continetal part of the USA 
    'BOUNDS': ((20, 50), (-127, -64)),

    # Output map colors
    'WATER_COLOR': '#9db8d3',
    'LAND_COLOR': '#f8f7f0',

    'OUTPUT_FILENAME': 'map.svg'
})

# Quadtree clustering parameters
CLUSTER = Bunch({
    # Change this boolean value to use maximum-depth stop criteria
    'BY_DEPTH': True,

    # Cell pattern for splitting (i.e. 2x2 for value of 2)
    'PATTERN': 2,

    'MAXIMUM_PER_GRID': 1000,
    'MAXIMUM_DEPTH': 5,
    'OUTPUT_FILENAME': 'clustered.csv'
})

# Word2Vec and logit parameters
CLASSIFY = Bunch({
    'W2V_FILENAME': 'w2v.dat',
    'TFIDF_FILENAME': 'vocabulary.dat',
    'LOGIT_FILENAME': 'logit.dat',
    'CV_ACCURACY_FILENAME': 'accuracy.txt',
    'F1_FILENAME': 'pre_rec_fm.txt',
    'ROC_FILENAME': 'roc.svg',
    'OUTPUT_FILENAME': 'predicted.csv',
    
    # Word2Vec learning paramters
    'W2V_MIN_COUNT': 1,
    'W2V_SIZE': 300,
    'W2V_WINDOW': 7,
    'W2V_WORKERS': 4,

    
    # Change this to use only cosine distance as criteria
    # of words similarity
    'USE_HYBRID': True,
    
    # Similarity thresholds for words eliminating
    'COSINE_THRESHOLD': 0.99,
    'JACCARD_THRESHOLD': 0.95,

    # Size of test part for logit learning
    'TEST_SIZE': 0.2,
    'CROSS_VALID_K': 5,
    'RANDOM_SEED': 42
})


# CSV input/output file column naming convention
CSV = Bunch({
    'INPUT' : Bunch({
        'INDEX': 'index_tweet',
        'TEXT': 'text',
        'LATITUDE': 'geocoordinate0',
        'LONGITUDE': 'geocoordinate1',
    }),

    'OUTPUT' : Bunch({
        'INDEX': 'index_tweet',
        'TEXT': 'text',

        'ACTUAL_GRID': 'ActualGrid',
        'PREDICTED_GRID': 'PredictedGrid',

        'ACTUAL_LATITUDE': 'ActualLatitude',
        'ACTUAL_LONGITUDE': 'ActualLongitude',

        'PREDICTED_LATITUDE': 'PredictedLatitude',
        'PREDICTED_LONGITUDE': 'PredictedLongitude',
        'ERROR': 'GreatCircle(km)'
    })
})