# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:38:00 2019

@author: gpinn
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import gensim 
import logging
import numpy as np
import json
from nltk import word_tokenize
from keras.models import Model
from keras.layers import Input, LSTM, Dense
#configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
tf.enable_eager_execution()

#isolate steps into functions
#load information
#download your datasets - two files containing questions & rationales in json files
aquarat_train = json.load(open("C:\\Users\\gpinn\\.spyder-py3\\AQuA-master\\AQuA-master\\train_without_debtor.tok.json"))
aquarat_test = json.load(open("C:\\Users\\gpinn\\.spyder-py3\\AQuA-master\\AQuA-master\\test.tok.json"))

list_questions = []
list_rationales = []
sentence_questions = []
sentence_rationales = []
vectorized_questions = []
vectorized_rationales = []

#preprocess the data - i.e. tokenize and vectorize.
for i in range(0, len(aquarat_train)):
    #find a way to replace the numbers with a preprocess - approved token
    #tokenize the items
    list_questions.append(gensim.utils.simple_preprocess(aquarat_train[i]['question']))
    list_rationales.append(gensim.utils.simple_preprocess(aquarat_train[i]['rationale']))
    #sentence_questions.extend(gensim.utils.simple_preprocess(aquarat_train[i]['question']))
    #sentence_rationales.extend(gensim.utils.simple_preprocess(aquarat_train[i]['rationale']))
   
"""
for i in range(0, 10):
    print(list_questions[i])

"""

model_questions = gensim.models.Word2Vec (list_questions, size=150, window=10, min_count=1, workers=10)
model_questions.train(list_questions,total_examples=len(list_questions),epochs=10)

#model_answers = gensim.models.Word2Vec (list_rationales, size=150, window=10, min_count=2, workers=10)
#model_answers.train(list_rationales,total_examples=len(list_rationales),epochs=10)

"""
So you can actually get vectors from word2vec, but this is in the form of a 1d numpy array
convert it to a vector via the following method - vector summary, root mean square, sentence vector
"""

#convert words in each sentence to its vectors
#iterate over all items in list_questions
for i in range(0, len(list_questions)):
    #for each question, we need to iterate over each word
    vectorized_words_in_question = []
    for j in range(0, len(list_questions[i])):
        #for each word, we need the matrix from word2vec
        vectorized_words_in_question.append(model_questions[(((list_questions[i])[j]))])

    #now append the list to the list of vectorized questions
    vectorized_questions.append(vectorized_words_in_question)
    
print(vectorized_questions[1])

"""
print(list_questions[25])

w1 = "plan"
wlist = ["You", "have", "ten", "apples", "and", "your", "friend"]
print(model_questions[w1])
#print(model_questions["plan"])
"""
