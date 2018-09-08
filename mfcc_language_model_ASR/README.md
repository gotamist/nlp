Here, we train a end_to_end speech recognition model.

Mel Frequency Cepstral Coefficients are used.

The network uses one convolutional layer and a deep RNN (5 layers).
GRUs were found to perform better than LSTMs and SimpleRNNs.

This version is trained on the full LibriSpeech dataset (360 hours )
which can be obtained from http://openslr.org/12/ .  It's possible to
achieve much smaller validation loss (under 50) than the dev-clean
dataset found on the same page.

A language model is also to be added.