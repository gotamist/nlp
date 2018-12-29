# Sentiment Analysis in documents

The notebook here shows how a neural network can be trained for
sentiment analysis. This is done without a predefined dictionary.
Words connoting positive and negative words are inferred from the
corpus itself so that the the dictionary is well suited to the corpus.

## Creation of the dictionary
In this supervised training task, positive and negative words are
identified based on the relative frequency with which they occur in
positive and negative movie reviews in the training corpus
respectively.  The same idea can be used to idetinfy dictionaries of
various kinds (to detect, obfuscation, say). Of course, the training
in this method needs a labeled corpus.

## Training

Backprop is implemented from scratch and no programming framework or
optimizer is used.

The reviews in the test set are then classified.  As a side effect of
the training, it is possible to find words that are closest in meaning
to any given word by examining the weight vectors learned for the
words and using a cosine distance between them as a similarity
measure.  


t-SNE is invoked for a visualization of the words in two
dimensions. Positive and negative words are seen to form two clusters.