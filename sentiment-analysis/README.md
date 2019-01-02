# Sentiment Analysis

The scripts in this folder show various methods of supervised learning
of sentiment.  This could be by using classical machine learning
algorithms, neural networks or by using some RNN (for example, an
LSTM) to process labelled documents.

The data is the [IMDB Moview Reviews sentiment classification
dataset](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification)
which is a set of 25,000 movie reviews that comes with Keras.  The
test set is 1,000 of these reviews and the remaining 24,000 are used
for training.

The best performance here is with the RNN (88%), which beats a
Gradient Boosted Classifer (79%) and Naïve Bayes (73%) on the same
dataset, although than can vary for problems.  None of these needs a
manually constructed dictionary