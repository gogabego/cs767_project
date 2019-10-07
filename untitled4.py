from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range

from tensorflow.python.framework.ops import disable_eager_execution
#disable eager execution
disable_eager_execution()

#for the representation of the results
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset.
TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True

#generate data 
#we have this, just import it
aquarat_train = json.load(open("cs767_project/AQuA-master/AQuA-master/train_without_debtor.tok.json"))
aquarat_test = json.load(open("cs767_project/AQuA-master/AQuA-master/test.tok.json"))

#vectorize data
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
    vectorized_test_questions.append(vectorized_words_in_question)
    
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
    
# Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
# int is DIGITS.
#MAXLEN = DIGITS + 1 + DIGITS
#you already have max length for both q and r, see above.

# Explicitly set apart 10% for validation data that we never train over.
#we have this already, just load it and set it up separately
(x_train, x_val) = np.asarray(vectorized_questions), np.asarray(vectorized_test_questions)
(y_train, y_val) = np.asarray(vectorized_rationales), np.asarray(vectorized_test_rationales)

#see the model you're working with
#we need three dimensions
#print((x_train))

# -everything before this works

#reshape to work as per https://stackoverflow.com/questions/44704435/error-when-checking-model-input-expected-lstm-1-input-to-have-3-dimensions-but

x_train = np.reshape(x_train, (x_train.shape[0], 1))
x_val = np.reshape(x_val, (x_val.shape[0], 1))
y_train = np.reshape(x_train, (x_train.shape[0], 1))
y_val = np.reshape(y_val, (y_val.shape[0], 1))


print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)


# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
#we're going to try maxlen for questions for this one given it's input
#model.add(RNN(HIDDEN_SIZE, input_shape=(q_max_length, len(q_vocab))))
model.add(RNN(HIDDEN_SIZE, input_shape=(1, 10), return_sequences=True))
# As the decoder RNN's input, repeatedly provide with the last output of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(DIGITS + 1))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)
