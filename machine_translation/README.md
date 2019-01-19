# Introduction


In this notebook, we build a deep neural network that functions as part
of an end-to-end machine translation pipeline. The completed pipeline
will accept English text as input and return the French translation.

The training dataset is the `small_vocab` set of sentences in
[StatMT](http://www.statmt.org), dedicated to statistical machine translation. 

Sequence models are a great choice for this problem. The notebook
contains a hierarchy of RNN models.

1. A simple RNN model (Gated Recurrence Units ) which takes one-hot
encoded words as input, followed by a dense layer.

2. Same structure as above but using an embedded word vector as
input. This time the RNN implementation is an LSTM (long Short Term
Memory) cell.

3. Bidirectional RNNs (GRUs again) which account for
the fact that the probability of a word in a certain place depends not
only on the words before it, but also the ones after it.

4. An encoder-decoder model where each part contains a GRU and
finally, the output is passed through a dense layer and a softmax
activation.

5. A bidirectional encoder-decoder implementation that also embeds the
input word vectors.

Finally, a reality check on a test set, using
`sklearn.model_selection.train_test_split()`


## Setup

This project requires GPU acceleration to run efficiently.  I ran this
on AWS.

## References

- The mathematics for statistical machine translation:
parameter estimation, *Brown, Della Pietra, Della Pietra and Mercer*,
1993, Association for Computational Linguistics.