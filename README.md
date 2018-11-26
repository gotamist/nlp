# nlp
## Natural Language Processing

The projects in this folder relate to Natural Language Processing,
which is a beautiful area of application of machine learning that I
find really exciting to work on.  The problems in this field are not
new, and it has been an area of investigation by compputer scientists
since the 1960s, but recent advances in deep neural networks have
driven considerable advancement in the field.

The projects that were done as part of the Udacity Nanodegree are
named starting with a digit.

### Part of Speech tagging

The problem here is to look at a sentence and for each word, identify
the part of speech (POS) that it represents.  For example, if the
input is

"My friend ate the delicious cake quickly", the output should be

PRONOUN - NOUN - VERB -DETERMINANT(article)-ADJECTIVE-NOUN-ADVERB.

The harder cases where disambiguation is necessary (the same word may
play different roles in different contexts) necessitate something more
than the Most Frequent Class (MFC) tagger, although the MFC tagger is
still useful as a baseline.  Here, a Hidden Markov Model is used and
various regularizations are used to deal with the problem of data
sparsity. Further details in the README in the project directory.

### Machine translation

The problem of machine translation is well suited for sequence models.
The input here is an English sentence and the output is the sentence
translated into French.  The solution uses recurrent neural networks
(LSTM/GRU), word embedding, bidrectional RNN and an encoder-decoder
setup.  Further details in the README within the machine translation
directory.

Another example of use of sequence models can be seen in this example
of [image
captioning](https://github.com/gotamist/vision/tree/master/image_captioning)
where the input is an image and output is a brief description of the
image in words.  Much like the translation problem, an encoder-decoder
architecture is used there.  The difference is just that the encoder
would use a CNN to produce an embedding vector to feed into the
decoder which is a sequence model.


### Speech recognition

The problem addressed in this project is one of taking an audio signal
as input and producing the transcript as output. Training and testing
is done using the [LibriSpeech
dataset](http://www.openslr.org/12/). The audio signal is first
transformed into a spectrogram (a possibility is to convert this into
Mel Frequency Cepstral Coefficients).  The acoustic models are trained
using Connectionist Temporal Classification
([CTC](http://www.cs.toronto.edu/~graves/icml_2006.pdf)) as loss in an
optimization routine. The architecture consists of a 1-dimensional CNN
layer followed by 5 layers of birectionals GRUs followed by a
TimeDistributed dense layer with softmax activation.  Dropout is used
for reguarization and batch normalization adds stability.

### The machine where the models are trained. 

Finally, I'm including the details of [the multi-GPU machine that I
built](https://github.com/gotamist/other_machine_learning/tree/master/deep-learning-machine). The
computations were partially performed on this machine. Some of them
were also run on a cloud (AWS).

I've added an MIT license here for my work.  Respecting the Udacity
license for the content and project, I have have added Udacity's stock
license within each project's folder.
