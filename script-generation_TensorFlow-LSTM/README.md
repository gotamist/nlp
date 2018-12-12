# Script generation for the Simpsons.

The dataset contains scripts from 27 episodes of the Simpsons, made
available at
[Kaggle](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data). The
network learns from that, to become able to generate a new script
given a seed to start with. The model is built with TensorFlow.

## Architecture

Words are embedding using an embedding layer which is connected to a
multi-RNN cell built with TensorFlow (by stacking Basic LSTM cells).
This is then passed through a fully connected layer.

## Training

Training is in minibatches. The loss is a **sequence_loss**, which takes
in the logits (the corresponding softmax would be the probability
distribution over all words) and target words as arguments.  The
optimizer is *Adam*.  *Gradient clipping* has to be employed.

## Choosing words to output

For this problem, it's preferable to pick words using a probability
distribution over the most probable `n` words, rather than pick the
word that is the `argmax` in the probability distribution.

