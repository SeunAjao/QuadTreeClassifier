## Description

This project is aimed at predicting tweet geolocation by its content.


### Clustering

1. At the beginning it performs quadtree clustering of tweets by their latitude and longitude coordinates. 

2. Every tweet gets a grid label, which is a unique label of tree node it was clustered in.


### Reducing words space dimensionality

3. Then the dictionary of the words used in all tweets is embedded by word2vec CBOW model.

4. Embedded word vectors are used to determine words similarity using cosine distance, or combination of cosine distance and generalized jaccard similarity.

5. Most similar words from the step 3 are eliminated by replacing them to their synonyms.


### Learning model to predict location

6. Reduced tweets are embedded using TF-IDF model.

7. Vectors from the previous step are used in learning logit regression for predict grid labels from the step 1.


### Visualisation

8. Origin and predicted geocoordinates are used to draw tweet points on the map.

9. Origin (from clustering stage) and predicted (from learning + testing stage) grid labels are used to colorize tweet points. Green and red points correspond to correct and incorrect label predictions respectively.

## Settings
You can tune clustering, embedding, learning stages by changing following parameters in ```settings.py```:

- ```settings.CLUSTER.BY_DEPTH``` [Boolean]: use maximum-depth as stop condition of quadtree clustering

- ```settings.CLUSTER.PATTERN'``` [Int]: cell pattern for splitting step. Every split node will contains (Pattern x Pattern) number of child nodes.

- ```settings.CLUSTER.MAXIMUM_PER_GRID``` [Int]: maximum tweets per grid allowed (if default criteria is used)

- ```settings.CLUSTER.MAXIMUM_DEPTH``` [Int]: maximum tree depth allowed (if BY_DEPTH criteria is used)

- ```settings.CLASSIFY.USE_HYBRID``` [Boolean]: use both cosine and generalized jaccard similarity thresholds of word similarity as condition for eliminating similar words

- ```settings.CLASSIFY.*_THRESHOLD``` [Float]: similarity threshold for cosine/generalized jaccard distance, used in words eliminating stage

- ```settings.CLASSIFY.W2V*```: word2vec CBOW parameters. See gensim word2vec documentation for more detailed description.



## Usage

Run pip install -r requirements.txt to install the required modules for the application 

Input csv format must be formatted as following, using tab as separate character:
```
index_tweet text geocoordinate0 geocoordinate1
... ... ... ...
```
When geocoordinate0, geocoordinate1 are latitude, longitude respectively.

Run main file as python script, suppressing warnings:
```
python -W ignore main.py INPUT_FILE
```