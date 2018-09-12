#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:39:01 2018

@author: gotamist
"""
import numpy as np
from data_generator import AudioGenerator
import re
from itertools import chain
    
def levenshtein(seq1, seq2):  
    # Thanks for this function to Frank Hoffman at 
    # https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
#    print (matrix)
    return (matrix[size_x - 1, size_y - 1])
#print(levenshtein('test','teams')) #works!

def generate_corpus(desc_file):
    #outputs a list of sentences
    data_sentences = AudioGenerator()
    data_sentences.load_train_data(desc_file=desc_file)
    sentences = data_sentences.train_texts
    return sentences

def wordset_from_corpus(sent_list):
    word_list = []
    for sent in sent_list:
        word_list.append(sent.split())
    long_word_list = [word for word in chain.from_iterable(word_list)]
#    words = re.findall('\w+', long_string)
    return set(long_word_list)

#st = generate_corpus("./train_corpus.json")
train_words = wordset_from_corpus(st)
## use the lines below to generate the txt on which to train kenlm
## the arpa file will be generated from this
#with open('corpus_360_lines.txt', 'w') as filehandle:  
#    filehandle.writelines("%s\n" % sentence for sentence in st)    
    
#Test kenlm using the python module contributed to kenlm by Victor Chahuneau.
# pip install https://github.com/kpu/kenlm/archive/master.zip
# see more here https://github.com/kpu/kenlm   
#import kenlm
#model = kenlm.Model('corpus_360_lines.arpa')
#
#print(model.score('play the matter', bos = True, eos = True))
    
#st_small= generate_corpus("./small_train_corpus.json")
#with open('small_corpus_lines.txt', 'w') as filehandle:  
#    filehandle.writelines("%s\n" % sentence for sentence in st_small)
    
def get_neighborhood(string, wordset, distance):
    """Finds all words from a set of words that are within a specified Levenshtein
    Distance from a given string"""
    nbd = [word for word in wordset if levenshtein(string, word) <= distance ]
    
