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
1. governor: minister, appointed, appoint, resigned, senate, state, elected, republican,
2. heavy: metal, reactor, isotopes, such, paces, phenyl, synthesized, grindcore,
3. discovered: discovery, found, discoveries, remains, was, reached, recently, unknown,
4. states: united, state, us, arrhenius, america, congress, colonies, ribbon,
5. three: four, two, five, six, seven, one, eight, nine,
6. mean: appropriate, weighted, portmanteau, meanings, skewness, sparknotes, harmonic, reciprocal,
7. pope: papal, pius, papacy, legates, managed, antipope, pontiff, xxiii,
8. grand: prix, slam, myself, theft, actium, lodge, oscillates, abimelech,
9. egypt: spain, suez, prospered, tutankhamun, tunisia, bahri, vater, malta,
10. ice: iceberg, frozen, mayfield, colder, cream, glacial, glacier, skates,
11. resources: resource, extremes, arable, pastures, petroleum, finkel, kaolin, bauxite,
12. between: both, and, which, anisotropic, there, a, of, along,


Some common embeddings are `word2vec` and `GloVe`. Skip-gram and
Continuous Bag of Words (CBOW) are two common ways to train word2vec.
Here, I'm including a notebook showing the word2vec embedding using
Skip-grams.

