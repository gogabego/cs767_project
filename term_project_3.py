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
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation
#configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#tf.enable_eager_execution()

#isolate steps into functions
#load information
#download your datasets - two files containing questions & rationales in json files
aquarat_train = json.load(open("cs767_project/AQuA-master/AQuA-master/train_without_debtor.tok.json"))
aquarat_test = json.load(open("cs767_project/AQuA-master/AQuA-master/test.tok.json"))

list_questions = []
list_rationales = []
test_questions = []
test_rationales = []
vectorized_questions = []
vectorized_rationales = []
vectorized_test_questions = []
vectorized_test_rationales = []
q_vocab = []
r_vocab = []
vectorized_qv = {}
vectorized_rv = {}

#we need some preprocessing for numbers in this 
#try custom preprocess filters
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation]
#preprocess the data - i.e. tokenize and vectorize.
#for i in range(0, len(aquarat_train)):
for i in range(0, 10):
    #find a way to replace the numbers with a preprocess - approved token
    #tokenize the items
    list_questions.append(preprocess_string((aquarat_train[i]['question']), CUSTOM_FILTERS))
    list_rationales.append(preprocess_string((aquarat_train[i]['rationale']), CUSTOM_FILTERS))
    test_questions.append(preprocess_string((aquarat_test[i]['question']), CUSTOM_FILTERS))
    test_rationales.append(preprocess_string((aquarat_test[i]['rationale']), CUSTOM_FILTERS))
    #get joint lists for vocab
    joint_questions = list_questions + test_questions
    joint_rationales = list_rationales + test_rationales
   
#model for all text
model_question = gensim.models.Word2Vec (joint_questions, size=150, window=10, min_count=1, workers=10)
model_question.train(joint_questions,total_examples=len(joint_questions),epochs=10)
model_rationale = gensim.models.Word2Vec (joint_rationales, size=150, window=10, min_count=1, workers=10)
model_rationale.train(joint_rationales,total_examples=len(joint_rationales),epochs=10)

#get weights of the models
mq_weight = model_question.wv.syn0
mr_weight = model_rationale.wv.syn0
qm_size, qme_size = mq_weight.shape
qr_size, qre_size = mr_weight.shape

#convert words in each sentence to its vectors
#iterate over all items in list_questions
for i in range(0, len(list_questions)):
    #for each question, we need to iterate over each word
    #first part is for vocab, second is for vectorization
    q_vocab.extend(list_questions[i])
    vectorized_words_in_question= []
    for j in range(0, len(list_questions[i])):
        #for each word, we need the matrix from word2vec
        vectorized_words_in_question.append(model_question[(((list_questions[i])[j]))])

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
        vectorized_words_in_rationale.append(model_rationale[(((list_rationales[i])[j]))])

    #now append the list to the list of vectorized questions
    vectorized_rationales.append(vectorized_words_in_rationale) 
    
#iterate over all items in test_questions
for i in range(0, len(test_questions)):
    #for each question, we need to iterate over each word
    #first part is for vocab, second is for vectorization
    q_vocab.extend(test_questions[i])
    vectorized_words_in_question= []
    for j in range(0, len(test_questions[i])):
        #for each word, we need the matrix from word2vec
        vectorized_words_in_question.append(model_question[(((test_questions[i])[j]))])

    #now append the list to the list of vectorized questions
    vectorized_test_questions.extend(vectorized_words_in_question)
    
#iterate over all items in list_rationales
for i in range(0, len(test_rationales)):
    #for each question, we need to iterate over each word
    #first part is for vocab, second is for vectorization
    r_vocab.extend(test_rationales[i])
    vectorized_words_in_rationale= []
    for j in range(0, len(test_rationales[i])):
        #for each word, we need the matrix from word2vec
        vectorized_words_in_rationale.append(model_rationale[(((test_rationales[i])[j]))])

    #now append the list to the list of vectorized questions
    vectorized_test_rationales.append(vectorized_words_in_rationale)
    
#delete duplicates and vectorize dictionary
q_vocab = (list(set(q_vocab)))
q_vocab.sort()
r_vocab = (list(set(r_vocab)))
r_vocab.sort()
#get length
qv_size = len(q_vocab)
rv_size = len(r_vocab)
#get max lengths
q_max_length = max([len(text) for text in list_questions])
r_max_length = max([len(text) for text in list_rationales])

#assign dictionary values - key is word, value is array of vectors
for i in range(0, len(q_vocab)):
    vectorized_qv[q_vocab[i]] = model_question[q_vocab[i]]
    
#do the same for the rationale vocabulary   
for i in range(0, len(r_vocab)):
    vectorized_rv[r_vocab[i]] = model_rationale[r_vocab[i]]

"""   
#create the arrays
encoder_input_data = np.zeros(
    (len(list_questions), q_max_length, qv_size),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(list_rationales), r_max_length, rv_size),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(list_rationales), r_max_length, rv_size),
    dtype='float32')
   
#Time Step for LSTM Layer
for pair_text_idx, (vectorized_q, vectorized_r) in enumerate(zip(list_questions, list_rationales)):
    for timestep, word in enumerate(vectorized_q):
        encoder_input_data[pair_text_idx, timestep, vectorized_qv[word]] = 1.
    # decoder_target_data is ahead of decoder_input_data by one timestep
    for timestep, word in enumerate(target_text):
        decoder_input_data[pair_text_idx, timestep, vectorized_rv[word]] = 1.
        if timestep > 0:
            # decoder_target_data will be ahead by one timestep（LSTM は前タイムステップの隠れ状態を現タイムステップの隠れ状態に使う）
            # decoder_target_data will not include the start character.
            decoder_target_data[pair_text_idx, timestep - 1, vectorized_rv[word]] = 1.
"""

#building dictionaries
def question_word2idx(word):
  return model_question.wv.vocab[word].index
def question_idx2word(idx):
  return model_question.wv.index2word[idx]
def rationale_word2idx(word):
  return model_rationale.wv.vocab[word].index
def rationale_idx2word(idx):
  return model_rationale.wv.index2word[idx]

#fill the existing input / target arrays with values 
#start with encoder
blank = 0
print('\nPreparing the data for LSTM...')
encoder_input_data = np.zeros([len(list_questions), q_max_length, blank], dtype=np.int32)
decoder_input_data = np.zeros([len(list_rationales), r_max_length, blank], dtype=np.int32)
decoder_target_data = np.zeros([len(list_rationales), r_max_length, blank], dtype=np.int32)
  
for t, rationale in enumerate(list_rationales):
  for l, word in enumerate(rationale):
    print(word)
    decoder_input_data[t, l] = rationale_word2idx(word)
  decoder_target_data[t] = rationale_word2idx(rationale[-1])
  
  
for i, sentence in enumerate(list_questions):
  for t, word in enumerate(sentence[:-1]):
    encoder_input_data[i, t] = question_word2idx(word)
    
print('encoder input shape:', encoder_input_data.shape)
print('decoder input shape:', decoder_input_data.shape)
print('decoder output shape:', decoder_target_data.shape)

NUM_HIDDEN_UNITS = 256 # NUM_HIDDEN_LAYERS
BATCH_SIZE = 64
NUM_EPOCHS = 10

#Encoder Architecture
#encoder_inputs = Input(shape=(None, qv_size))
encoder_inputs = Input(shape=(58, ))
encoder_embed = Embedding(input_dim=qm_size, output_dim=qme_size, weights=[mq_weight])(encoder_inputs)
encoder_lstm = LSTM(units=NUM_HIDDEN_UNITS, return_sequences=True, return_state=True)
# x-axis: time-step lstm
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c] # We discard `encoder_outputs` and only keep the states.

#Decoder Architecture
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
#decoder_inputs = Input(shape=(None, rv_size))
decoder_inputs = Input(shape=(137, ))
decoder_embed = Embedding(input_dim=qr_size, output_dim=qre_size, weights=[mr_weight])(decoder_inputs)
decoder_lstm = LSTM(units=NUM_HIDDEN_UNITS, return_sequences=True, return_state=True)
# x-axis: time-step lstm
decoder_outputs, de_state_h, de_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states) # Set up the decoder, using `encoder_states` as initial state.
#print(rv_size)
decoder_softmax_layer = Dense(decoder_LSTM, activation='softmax')
print(type(decoder_softmax_layer))
decoder_outputs = decoder_softmax_layer(decoder_outputs)

#Encoder-Decoder Architecture
# Define the model that will turn, `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy") # Set up model

print(model.summary())

#change the vectorized components to arrays

#reshape the data
#encoder_input_data = np.reshape(encoder_input_data,(1, 10, 58))
#decoder_input_data = np.reshape(decoder_input_data,(1, 10, 137))
#works up to this point
model.fit(x=[encoder_input_data, decoder_input_data], y=decoder_target_data,
          batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2) 
