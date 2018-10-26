# nlp
## Natural Language Processing

The projects in this folder relate to Natural Language Processing, which is a beautiful area of application of machine learning that I found really exciting to work on.  
The projects that were done as part of the Udacity Nanodegree are named starting with a digit. 

### Part of Speech tagging
The problem here is to look at a sentence and for each word, identify the part of speech (POS) that it represents.  
For example, if the input is "My friend ate the delicious cake quickly", the output should be PRONOUN - NOUN - VERB - DETERMINANT(article)-ADJECTIVE-NOUN-ADVERB. Sometimes, the same word may correspond to different parts of speech.
For example, "Will will never refuse to permit me to apply for a refuse permit".  Here "will"(NOUN/VERB), "refuse"(VERB/NOUN) and "permit"(VERB/NOUN) appear in multiple roles. The goal here is to teach a machine to identify that accurately. Here, a Hidden Markov Model (HMM) is used to make the prediction. A simple MFC tagger (where a word in the test set is simply assigned its most frequent POS tag in the training set) is used as a reference to show the significant improvement in accuracy from using the HMM.

This type of problem always has to deal with the issue of unseen words (where a word appearing in the test set was never in the training set) or unseen word-tag pairs ("refuse" may only ever have appeared in the training as a verb and never as a noun). Laplace Smoothing and its generalization, *add-k smoothing* are also implemented.  Other regularizations like Baum-Welch reestimation and back-off smoothing are also shown.  The HMM model is used to make predictions on unseen sentences. The model's accuracy can be improved by training the model using the Brown corpus with the full NLTK tagset. 


## The machine
Finally, I'm including the details of the multi-GPU machine that I built for performing these projects.  

I've added an MIT license here for my work.  Respecting the Udacity license for the content and project, I have have added Udacity's stock license within each project's folder.
