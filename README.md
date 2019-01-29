# NLP
## Natural Language Processing

The projects in this folder relate to Natural Language Processing,
which is a beautiful area of application of machine learning that I
find really exciting to work on.  The problems in this field are not
new, and it has been an area of investigation by computer scientists
since the 1960s, but recent advances in deep neural networks have
driven considerable advancement in the field.

The directories here contain projects covering various topic and
problems.  Below is a brief overview of some of the projects.

### Sentiment analysis

In this folder, you'll find models to predict sentiment
(postive/negative). Training is done on IMDB reviews.  I've shown a
performance comparison of training using Gradient Boosting, a vanilla
neural net and a recurrent neural net.  Please note that **none of
these uses a manually constructed dictionary.** Dictionaries have to
be constructed with care so that they are meaningful for the corpus at
hand.  [Loughran and McDonald,
2016](https://onlinelibrary.wiley.com/doi/abs/10.1111/1475-679X.12123)
and [El-Haj et al
2018](https://www.smurfitschool.ie/media/businessschool/profileimages/docs/generalpdfs/01%20-%20Stephen%20Young%20Conference%20Paper.pdf)
contain references to studies that many common collections of
sentiment, for example are not suited for financial analysis and how
to construct one that is more suitable.  As a simple example, "crude"
is not a negative word in the text of a company in the energy sector.
Nor is there a reason to consider the words "liability" and
"depreciation" to be negative in 10-K filings of firms - these are
common words.

### Training word embeddings

Here, you'll find an example script for training word2vec on any
corpus.  Word2vec and GloVe are two commonly used word embeddings.
Word embeddings are often required for doing textual analysis in
signal research.  Again word embeddings that are specific to a corpus
(broker reports corpus or filings corpus). [Davies and Tama-Sweet,
2012](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1911-3846.2011.01125.x)
and others also find systematic differences in linguistic features
across different corpora - filings, conference calls, etc.  It is
quick to train word2vec that is suited for a corpus if needed to get
better results.


### Part of Speech Tagging (POST)

The problem here is to look at a sentence and for each word, identify
the part of speech (POS) that it represents.  For example, if the
input is

"They refuse to permit us to obtain the new refuse permit", the output should be

PRONOUN-VERB-PREPO-VERB-PRONOUN-PREPO-VERB-DETERMINANT(article)-ADJ-NOUN-NOUN.

The harder cases where disambiguation is necessary (the same word may
play different roles in different contexts) necessitate something more
than the Most Frequent Class (MFC) tagger, although the MFC tagger is
still useful as a baseline.  Here, a Hidden Markov Model is used and
various regularizations are used to deal with the problem of data
sparsity. Further details in the README in the project directory.

The use of different parts of speech provides a window into complexity
and sometimes, obfuscative intent.

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

### Automatic Speech recognition

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

I've added an MIT license here for my work.  If the directory contains
a project that was done as part of the Udacity NLP Nanodegree's
project requirement, that is indicated in within the
folder. Respecting the Udacity license for the content and project, I
have added Udacity's stock license within folder where appropriate.
