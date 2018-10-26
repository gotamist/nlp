# nlp
## Natural Language Processing

The projects in this folder relate to Natural Language Processing,
which is a beautiful area of application of machine learning that I
find really exciting to work on.  The problems in this field are not
new, and it has been an area of investigation by compputer scientists
since the 1960s, but recent advances in deep neural networks have
driven considerable advancement in the field.

The projects that were done as part
of the Udacity Nanodegree are named starting with a digit.

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
sparsity.


### The machine
Finally, I'm including the details of the multi-GPU machine that I built for performing these projects.  

I've added an MIT license here for my work.  Respecting the Udacity
license for the content and project, I have have added Udacity's stock
license within each project's folder.
