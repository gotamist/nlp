# Latent Dirichlet Allocation applied to Topic Modeling

We would like to classify the documents in a corpus as discussing one
of several (in this example, 10) rather than simply identify a
document by the words it has (either as a bag of words or in the
TF-IDF form).

Here, Latent Dirichlet Allocation, as described in [Blei et al,
2003](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) is used
to extract the topics.

A document may be associated with more than one topic and it's
possible to identify the relative importance of each topic in a
document.

Here, the LDA function is used from the gensim package. In particular,
it allows for multicore training (I've run it on 6 cores).  The LDA
can be fed the original documents in either representation - BoW or
TF-IDF.  Note that the topics produced are different in the two cases.
Topics are identified from the important words that appear in the
topic and it is upto us to provide an interpretation of what each
identified topic represents.
