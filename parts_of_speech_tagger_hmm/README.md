# Part of Speech tagging using Hidden Markov Models

The problem here is to look at a sentence and for each word, identify
the part of speech (POS) that it represents.

For example, if the input is
"My friend ate the delicious cake quickly", the output should be
PRONOUN - NOUN - VERB - DETERMINANT(article)-ADJECTIVE-NOUN-ADVERB.

Sometimes, the same word may correspond to different parts of speech.
For example, "Will will never refuse to permit me to apply for a
refuse permit"..  Here "will"(NOUN/VERB), "refuse"(VERB/NOUN) and
"permit"(VERB/NOUN) appear in multiple roles. POS tagging must solve this __disambiguation__ task. 

The goal is to teach a machine to identify that accurately. Here, a
__Hidden Markov Model__ (HMM) is used to make the prediction.  The
[Pomegranate library](https://pomegranate.readthedocs.io/en/latest/)
is used to construct the HMM.  A simple most frequent class (MFC)
tagger where a word in the test set is simply assigned its most
frequent POS tag in the training set, is used as a reference to show
the significant improvement in accuracy from using the HMM. The
accuracy of these N-gram models improves if we can use higher N,
(trigrams are better than bigrams, which in turn, are better than
unigrams).  But then the *data sparsity problem* becomes more and more
severe with rising N.

This type of problem always has to deal with the issue of __unseen words__
(where a word appearing in the test set was never in the training set)
or __unseen word-tag pairs__ ("refuse" may only ever have appeared in the
training as a verb and never as a noun).  Laplace Smoothing and its
generalization, *add-k smoothing* are also implemented.  Other
__regularizations__ like Baum-Welch re-estimation and back-off smoothing
are shown.  The HMM model is used to make predictions on unseen
sentences.  The model's accuracy can be improved by training the model
using the Brown corpus with the full NLTK tagset.

### References

- Jurafsky and Martin, *Speech and Language Processing*, 2017