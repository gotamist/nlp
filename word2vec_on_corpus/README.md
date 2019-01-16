# Word embeddings - Word2vec

There may be tens of thousands of words in a corpus.  If we can embed
these words into a vector space of, say 200 or 300 words, then not
only can we surmount the enormous computations inefficiency we would
have from one-hot encoding, we can use it to look for neighbors of
words within the embedding space.  The embeddings can also be projected
to a further smaller space using PCA or t-SNE, which is useful for
visualization.

Here are some examples of neighborhoods from the training is this
notebook.

1. more: than, less, are, have, most, those, much, all,
2. between: both, and, which, anisotropic, there, a, of, along,
3.  are: there, these, have, all, or, and, other, most,
4. states: united, state, us, arrhenius, america, congress, colonies, ribbon,
5. brother: murdered, father, grandson, son, downy, grandmother, rumpus, eldest,
6. mean: distance, harmonic, aasen, nth, similarly, nameless, ramadan, jansen,
7. pope: papal, pius, papacy, legates, managed, antipope, pontiff, xxiii,
8. grand: prix, slam, myself, theft, actium, lodge, oscillates, abimelech,
9. egypt: spain, suez, prospered, tutankhamun, tunisia, bahri, vater, malta,
10. alternative: complementary, integrative, medicine, illich, californians, fincher, edta, placebo,
11. resources: resource, extremes, arable, pastures, petroleum, finkel, kaolin, bauxite,

Some common embeddings are `word2vec` and `GloVe`. Skip-gram and
Continuous Bag of Words (CBOW) are two common ways to train word2vec.
Here, I'm including a notebook showing the word2vec embedding using
Skip-grams.

