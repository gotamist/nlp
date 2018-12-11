# Script generation for the Simpsons.

The dataset contains scripts from 27 episodes of the Simpsons, made
available at
[Kaggle](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data). The
network learns from that, to become able to generate a new script
given a seed to start with.

## Architecture
Multi-RNN cell built with TensorFlow (by stacking Basic LSTM cells). 

