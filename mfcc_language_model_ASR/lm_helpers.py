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
from textblob import TextBlob as tb
import kenlm
import panphon.distance
from fuzzy import DMetaphone

def create_DMetaphone_list(wordset):
    dm = DMetaphone(5)
    codedict = {}
    codeset = set()
    for word in wordset:
        strings2 = [  str(code)[2:-1] for code in dm(word) if code is not None ]
        codedict[word] = strings2 
        codeset.update( set(strings2) )
    return codedict, codeset


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
#train_words = wordset_from_corpus(st)
#valid_words = wordset_from_corpus(generate_corpus("./valid_corpus.json") )  
#unseen_words =[word for word in valid_words if word not in train_words] 
#hmmm...733 such unseen words (not many are proper nouns).  
# Need a larger wordset than just the train_words (try nltk)
    
#st_small= generate_corpus("./small_train_corpus.json")
#with open('small_corpus_lines.txt', 'w') as filehandle:  
#    filehandle.writelines("%s\n" % sentence for sentence in st_small)
    
def get_neighborhood(string, wordset, distance):
    """Finds all words from a set of words that are within a specified Levenshtein
    Distance from a given string"""
    nbd = [word for word in wordset if levenshtein(string, word) <= distance ]
    return set( nbd )

def dolgopolsky_neighborhood(string, wordset, distance):
    """Finds all words from a set of words that are within a specified Levenshtein
    Distance from a given string"""
    dst = panphon.distance.Distance()
    nbd = [word for word in wordset if dst.dogol_prime_distance(string, word) <= distance ]
    return set( nbd )    

#nbd = get_neighborhood('helium', train_words, 2) #tested, 5 words found
#nbd = get_neighborhood('helium', english, 2) #tested, 43 words found
#sample_true = 'far up the lake eighteen miles above the town the eye of this cheerful camp follower of booms had spied out a graft'
#sample_input_sent = 'far ut the lake eightteen mils abo the town to ey of dis cherple can flolowor o bons had xpide ut a graft'
#sample_blob='far ut the lake eighteen miss ago the town to by of dis chere can follower o bons had side ut a graft'
#inp = input_sent.split()

#kenmodel = kenlm.Model('corpus_360_lines.arpa')
#ken5model = kenlm.Model('5_gram_corpus_360.binary')


def lm_predict(input_sentence, dictionary, lmodel): #input is a string
    #assumes that the output of the DNN is of the right length
    """this function keeps adding words that maximize the probability of sentence
    all the way from the beginning until the new word"""
    inp = input_sentence.split()
       
    #construct the first trigram
    #Note that the shortest sentence in this dataset has three words

    nbd0 = get_neighborhood( inp[0], dictionary, 2)
    nbd1 = get_neighborhood( inp[1], dictionary, 2)
    nbd2 = get_neighborhood( inp[2], dictionary, 2)
    tg={}

    for first_word in nbd0:
        for second_word in nbd1:
            for third_word in nbd2:
                trigram = first_word+' '+second_word+' '+third_word
                tg[ trigram ]=lmodel.score(trigram, bos = True, eos = False)
    
    pred=max(tg, key=tg.get)
    
    for i in range(3,len(inp)):
        phrases={}
        nbd = [ inp[i] ] if inp[i] in dictionary else get_neighborhood( inp[i], dictionary, 2)
        nbd = get_neighborhood( inp[i], dictionary, 2)
        
        for word in nbd:
            candidate=pred+' '+word
            phrases[ candidate ]=lmodel.score( candidate, bos = True, eos = False)
        pred=max(phrases, key=phrases.get)
    return pred    
# 'far to the lake eighteen pins so the town to ku of dis cheaply can o fans had pile ku a graft'    
    
def trigram_predict(input_sentence, train_dictionary, predict_dictionary, lmodel, radius=1.5): #input is a string
    #assumes that the output of the DNN is of the right length
    inp = input_sentence.split()
    
       
    #construct the first trigram
    #Note that the shortest sentence in this dataset has three words
    
    nbd0 = [ inp[0] ] if inp[0] in train_dictionary else get_neighborhood( inp[0], predict_dictionary, radius)
    nbd1 = get_neighborhood( inp[1], predict_dictionary, radius)
    nbd2 = get_neighborhood( inp[2], predict_dictionary, radius)
    tg={}

    for first_word in nbd0:
        for second_word in nbd1:
            for third_word in nbd2:
                trigram = first_word+' '+second_word+' '+third_word
                tg[ trigram ]=lmodel.score(trigram, bos = True, eos = False)
    
    pred=max(tg, key=tg.get)
    output = pred.split()
    
    for i in range(3,len(inp)):
        phrases={}
        nbd = [ inp[i] ] if inp[i] in train_dictionary else get_neighborhood( inp[i], predict_dictionary, radius)
#        nbd = get_neighborhood( inp[i], dictionary, 2)
        
        for word in nbd:
            candidate=output[-2]+' '+output[-1]+' '+word
            phrases[ word ]=lmodel.score( candidate, bos = False, eos = False)
        next_word=max(phrases, key=phrases.get)
        output.append( next_word )
        pred=pred+' '+next_word
        
    return pred        

# 'far to the lake eighteen miles above the town to be of dis cheaply can can o one had side of a graft'
def cumul_sweep(input_sentence, intermediate, dictionary):
    inp = input_sentence.split()
    inter=intermediate.split()
    
    for i in range(3,len(inp)):
        phrases={}
        nbd = get_neighborhood( inter[i], dictionary, 2) 
        u_nbd=nbd.union( get_neighborhood( inp[i], dictionary, 2) )   
        for word in u_nbd:
            candidate=pred+' '+word
            phrases[ word ]=lmodel.score( candidate, bos = False, eos = False)
        next_word=max(phrases, key=phrases.get)
        pred=pred+' '+next_word       
    return pred  
    
    
def trigram_dolgopolsky_predict(input_sentence, train_dictionary, predict_dictionary, lmodel, radius=1.5): #input is a string
    #assumes that the output of the DNN is of the right length
    inp = input_sentence.split()
    
    #construct the first trigram
    #Note that the shortest sentence in this dataset has three words
    #for the second word, use bigram prob from kenlm
    
    nbd0 = [ inp[0] ] if inp[0] in train_dictionary else dolgopolsky_neighborhood( inp[0], predict_dictionary, radius)
    nbd1 = dolgopolsky_neighborhood( inp[1], predict_dictionary, radius)
    nbd2 = dolgopolsky_neighborhood( inp[2], predict_dictionary, radius)
    tg={}

    for first_word in nbd0:
        for second_word in nbd1:
            for third_word in nbd2:
                trigram = first_word+' '+second_word+' '+third_word
                tg[ trigram ]=lmodel.score(trigram, bos = True, eos = False)
    
    pred=max(tg, key=tg.get)
    output = pred.split()
    
    for i in range(3,len(inp)):
        phrases={}
        nbd = [ inp[i] ] if inp[i] in train_dictionary else dolgopolsky_neighborhood( inp[i], predict_dictionary, radius)
        for word in nbd:
            candidate=output[-2]+' '+output[-1]+' '+word
            phrases[ word ]=lmodel.score( candidate, bos = False, eos = False)
        next_word=max(phrases, key=phrases.get)
        output.append( next_word )
        pred=pred+' '+next_word        
    return pred     
                
                
#test_dolgo=trigram_dolgoposlky_predict(sample_input_sent, train_dictionary=train_words, predict_dictionary=english, radius=1.5)  
#print( test_dolgo )     
       
def bigram_predict(input_sentence, train_dictionary, predict_dictionary, lmodel, radius=1.5): #input is a string
    #assumes that the output of the DNN is of the right length
    inp = input_sentence.split()
    
    nbd0 = [ inp[0] ] if inp[0] in train_dictionary else get_neighborhood( inp[0], predict_dictionary, radius)
    nbd1 = get_neighborhood( inp[1], predict_dictionary, radius)
    nbd2 = get_neighborhood( inp[2], predict_dictionary, radius)
    tg={}

    for first_word in nbd0:
        for second_word in nbd1:
            for third_word in nbd2:
                trigram = first_word+' '+second_word+' '+third_word
                tg[ trigram ]=lmodel.score(trigram, bos = True, eos = False)
    
    pred=max(tg, key=tg.get)
    output = pred.split()
    
    for i in range(3,len(inp)):
        phrases={}
        nbd = [ inp[i] ] if inp[i] in train_dictionary else get_neighborhood( inp[i], predict_dictionary, radius)
        
        for word in nbd:
            candidate=output[-1]+' '+word
            phrases[ word ]=lmodel.score( candidate, bos = False, eos = False)
        next_word=max(phrases, key=phrases.get)
        output.append( next_word )
        pred=pred+' '+next_word
        
    return pred        

def bigram_dolgopolsky_predict(input_sentence, train_dictionary, predict_dictionary, lmodel, radius=1.5): #input is a string
    #assumes that the output of the DNN is of the right length
    inp = input_sentence.split()
    
    #construct the first trigram
    #Note that the shortest sentence in this dataset has three words
    #for the second word, use bigram prob from kenlm
    
    nbd0 = [ inp[0] ] if inp[0] in train_dictionary else dolgopolsky_neighborhood( inp[0], predict_dictionary, radius)
    nbd1 = dolgopolsky_neighborhood( inp[1], predict_dictionary, radius)
    nbd2 = dolgopolsky_neighborhood( inp[2], predict_dictionary, radius)
    tg={}

    for first_word in nbd0:
        for second_word in nbd1:
            for third_word in nbd2:
                trigram = first_word+' '+second_word+' '+third_word
                tg[ trigram ]=lmodel.score(trigram, bos = True, eos = False)
    
    pred=max(tg, key=tg.get)
    output = pred.split()
    
    for i in range(3,len(inp)):
        phrases={}
        nbd = [ inp[i] ] if inp[i] in train_dictionary else dolgopolsky_neighborhood( inp[i], predict_dictionary, radius)
        
        for word in nbd:
            candidate=output[-1]+' '+word
            phrases[ word ]=lmodel.score( candidate, bos = False, eos = False)
        next_word=max(phrases, key=phrases.get)
        output.append( next_word )
        pred=pred+' '+next_word        
    return pred        
    
#for key, value in sorted(bg_scores.iteritems(), key=lambda (k,v): (v,k)):
#    print "%s: %s" % (key, value)
#newD = dict(sorted(bg.items(), key=operator.itemgetter(1), reverse=True)[:5])
    
#x = sorted(tg, key=tg.get, reverse=True)[:5]

## use the lines below to generate the txt on which to train kenlm
## the arpa file will be generated from this
#with open('corpus_360_lines.txt', 'w') as filehandle:  
#    filehandle.writelines("%s\n" % sentence for sentence in st)    
    
#Test kenlm using the python module contributed to kenlm by Victor Chahuneau.
# pip install https://github.com/kpu/kenlm/archive/master.zip
# see more here https://github.com/kpu/kenlm   
#import kenlm
#model = kenlm.Model('corpus_360_lines.arpa')
# print(model.score('play the matter', bos = True, eos = True))
    
