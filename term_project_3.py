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
vectorized_questions = []
vectorized_rationales = []
q_vocab = []
r_vocab = []
vectorized_qv = {}
vectorized_rv = {}

#we need some preprocessing for numbers in this 
#some way to include them, perhaps just writing them?

#preprocess the data - i.e. tokenize and vectorize.
for i in range(0, len(aquarat_train)):
    #find a way to replace the numbers with a preprocess - approved token
    #tokenize the items
    list_questions.append(gensim.utils.simple_preprocess(aquarat_train[i]['question']))
    list_rationales.append(gensim.utils.simple_preprocess(aquarat_train[i]['rationale']))
    #sentence_questions.extend(gensim.utils.simple_preprocess(aquarat_train[i]['question']))
    #sentence_rationales.extend(gensim.utils.simple_preprocess(aquarat_train[i]['rationale']))
   
#model for all text
model_question = gensim.models.Word2Vec (list_questions, size=150, window=10, min_count=1, workers=10)
model_question.train(list_questions,total_examples=len(list_questions),epochs=10)
model_rationale = gensim.models.Word2Vec (list_rationales, size=150, window=10, min_count=1, workers=10)
model_rationale.train(list_rationales,total_examples=len(list_rationales),epochs=10)

"""
#model for questions
model_questions = gensim.models.Word2Vec (list_questions, size=150, window=10, min_count=1, workers=10)
model_questions.train(list_questions,total_examples=len(list_questions),epochs=10)
#model for answers
model_answers = gensim.models.Word2Vec (list_rationales, size=150, window=10, min_count=2, workers=10)
model_answers.train(list_rationales,total_examples=len(list_rationales),epochs=10)
"""
"""
So you can actually get vectors from word2vec, but this is in the form of a 1d numpy array
convert it to a vector via the following method - vector summary, root mean square, sentence vector
"""

#convert words in each sentence to its vectors
#iterate over all items in list_questions
for i in range(0, len(list_questions)):
    #for each question, we need to iterate over each word
    #first part is for vocab, second is for vectorization
    q_vocab.extend(list_questions[i])
    vectorized_words_in_question= []
    for j in range(0, len(list_questions[i])):
        #for each word, we need the matrix from word2vec
        vectorized_words_in_question.append(model_questions[(((list_questions[i])[j]))])

    #now append the list to the list of vectorized questions
    vectorized_questions.append(vectorized_words_in_question)

#iterate over all items in list_rationales
for i in range(0, len(list_rationales)):
    #for each question, we need to iterate over each word
    #first part is for vocab, second is for vectorization
    r_vocab.extend(list_rationales[i])
    vectorized_words_in_rationale= []
    for j in range(0, len(list_rationales[i])):
        #for each word, we need the matrix from word2vec
        vectorized_words_in_rationale.append(model_rationales[(((list_rationales[i])[j]))])

    #now append the list to the list of vectorized questions
    vectorized_rationales.append(vectorized_words_in_rationale)    
    
#delete duplicates and vectorize dictionary
q_vocab = (list(set(q_vocab))).sort()
r_vocab = (list(set(r_vocab))).sort()

#assign dictionary values - key is word, value is array of vectors
for i in range(0, len(q_vocab)):
    vectorized_qv[q_vocab[i]] = model_questions[q_vocab[i]]
    
#do the same for the rationale vocabulary   
for i in range(0, len(r_vocab)):
    vectorized_rv[r_vocab[i]] = model_questions[r_vocab[i]]

"""
#Time Step for LSTM Layer
for pair_text_idx, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):

    for timestep, word in enumerate(input_text):
        encoder_input_data[pair_text_idx, timestep, inverse_input_vocab[word]] = 1.
    # decoder_target_data is ahead of decoder_input_data by one timestep
    for timestep, word in enumerate(target_text):
        decoder_input_data[pair_text_idx, timestep, inverse_target_vocab[word]] = 1.
        if timestep > 0:
            # decoder_target_data will be ahead by one timestep（LSTM は前タイムステップの隠れ状態を現タイムステップの隠れ状態に使う）
            # decoder_target_data will not include the start character.
            decoder_target_data[pair_text_idx, timestep - 1, inverse_target_vocab[word]] = 1.
"""

NUM_HIDDEN_UNITS = 256 # NUM_HIDDEN_LAYERS
BATCH_SIZE = 64
NUM_EPOCHS = 100
"""
#Encoder Architecture
encoder_inputs = Input(shape=(None, encoder_vocab_size))
encoder_lstm = LSTM(units=NUM_HIDDEN_UNITS, return_state=True)
# x-axis: time-step lstm
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c] # We discard `encoder_outputs` and only keep the states.

#Decoder Architecture
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_inputs = Input(shape=(None, decoder_vocab_size))
decoder_lstm = LSTM(units=NUM_HIDDEN_UNITS, return_sequences=True, return_state=True)
# x-axis: time-step lstm
decoder_outputs, de_state_h, de_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states) # Set up the decoder, using `encoder_states` as initial state.
decoder_softmax_layer = Dense(decoder_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

#Encoder-Decoder Architecture
# Define the model that will turn, `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy") # Set up model
"""
