#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = 'maxim'

import numpy as np
import gensim
import string
import json


from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation

from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file

#getting data
print('\nFetching the text...')
#download your datasets - two files containing questions & rationales in json files
aquarat_train = json.load(open("cs767_project/AQuA-master/AQuA-master/train_without_debtor.tok.json"))
aquarat_test = json.load(open("cs767_project/AQuA-master/AQuA-master/test.tok.json"))

list_questions = []
list_rationales = []
vectorized_questions = []
vectorized_rationales = []
vectorized_test_questions = []
vectorized_test_rationales = []
q_vocab = []
r_vocab = []
vectorized_qv = {}
vectorized_rv = {}

CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation]
#preprocess the data - i.e. tokenize and vectorize.
for i in range(0, len(aquarat_train)):
#for i in range(0, len(list_questions)):
    #find a way to replace the numbers with a preprocess - approved token
    #tokenize the items
    list_questions.append(preprocess_string((aquarat_train[i]['question']), CUSTOM_FILTERS))
    list_rationales.append(preprocess_string((aquarat_train[i]['rationale']), CUSTOM_FILTERS))
    #test_questions.append(preprocess_string((aquarat_test[i]['question']), CUSTOM_FILTERS))
    #test_rationales.append(preprocess_string((aquarat_test[i]['rationale']), CUSTOM_FILTERS))
    #get joint lists for vocab
    joint_questions = list_questions #+ test_questions
    joint_rationales = list_rationales #+ test_rationales

    
max_sentence_len = 0
for i in range(0, len(joint_questions)):
  if len(joint_questions[i]) > max_sentence_len:
    max_sentence_len = len(joint_questions[i])
    
    
#tokenizing data down to just words
print('\nPreparing the sentences...') 
sentences = joint_questions
print('Num sentences:', len(sentences))

#bulding the word2vec model
print('\nTraining word2vec...')
word_model = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=5, iter=100)
pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)
print('Checking similar words:')

#building dictionaries
def word2idx(word):
  return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]

#building data sets needed
print('\nPreparing the data for LSTM...')
train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
    train_x[i, t] = word2idx(word)
  train_y[i] = word2idx(sentence[-1])
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)

print('\nTraining LSTM...')
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size, return_sequences=True))
model.add(LSTM(units=emdedding_size, return_sequences=True))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

#building sample data
def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

#generating next word
def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word) for word in text.lower().split()]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature=0.7)
    word_idxs.append(idx)
  return ' '.join(idx2word(idx) for idx in word_idxs)

#trying to generate a sentence at the end of each epoch so you know what's going on
def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  texts = [
    'In the coordinate plane , points',
    'Mike took 5 mock tests before appearing for the GMAT . In each mock test he scored',
    'The first five numbers in a regular sequence are 4 , 10 , X , 46 , and 94 . What is',
    '5358 x 51 = ',
    'The cost of painting the whole surface area of a cube at the rate of'
  ]
  for text in texts:
    sample = generate_next(text)
    print('%s... -> %s' % (text, sample))

#fitting the model
model.fit(train_x, train_y,
          batch_size=128,
          epochs=20,
          callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
