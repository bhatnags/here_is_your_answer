# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:57:30 2019

@author: x219427
"""

import pandas as pd
import numpy as np
import json
from pprint import pprint
from pandas.io.json import json_normalize
import time, timeit
import os, sys
import keras
import tensorflow as tf


from keras.models import Model
from keras import layers
from keras.layers import Dense, Embedding, Input, LSTM, RepeatVector, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU


path = "C:\\Saurabh\\QA Systems"

with open(os.path.join(path, 'Data', 'train-v2.0.json')) as read_train:
    train = json.load(read_train)

with open(os.path.join(path, 'Data', 'dev-v2.0.json')) as read_test:
    test = json.load(read_test)


train_df_list = []
for i in range(len(train['data'])):
    
    for j in range(len(train['data'][i]['paragraphs'])):
        
        context = train['data'][i]['paragraphs'][j]['context']
        
        context = context.replace("''", '" ')
        context = context.replace("``", '" ')
        context = context.lower()
        
        for k in range(len(train['data'][i]['paragraphs'][j]['qas'])):
            
            question = train['data'][i]['paragraphs'][j]['qas'][k]['question']
            question = question.lower()
            
            is_impossible = train['data'][i]['paragraphs'][j]['qas'][k]['is_impossible']
            
            if is_impossible == False:
                
                for l in range(len(train['data'][i]['paragraphs'][j]['qas'][k]['answers'])):
                    
                    answer = train['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['text']
                    answer = answer.lower()
                    
                    answer_start = train['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['answer_start']
                    answer_end = answer_start + len(answer)
            else:  
                
                answer = ""
                answer_start = ""
                answer_end = ""
            train_df_list.append((context, question, is_impossible, answer, answer_start, answer_end))
train_df = pd.DataFrame(train_df_list, columns = ['Context', 'Question', 'Is_possible','Answer', 'Answer_Start', 'Answer_End'])


test_df_list = []
for i in range(len(test['data'])):
    
    for j in range(len(test['data'][i]['paragraphs'])):
        
        context = test['data'][i]['paragraphs'][j]['context']
        
        context = context.replace("''", '" ')
        context = context.replace("``", '" ')
        context = context.lower()
        
        for k in range(len(test['data'][i]['paragraphs'][j]['qas'])):
            
            question = test['data'][i]['paragraphs'][j]['qas'][k]['question']
            question = question.lower()
            
            is_impossible = test['data'][i]['paragraphs'][j]['qas'][k]['is_impossible']
            
            if is_impossible == False:
                
                for l in range(len(test['data'][i]['paragraphs'][j]['qas'][k]['answers'])):
                    
                    answer = test['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['text']
                    answer = answer.lower()
                    
                    answer_start = test['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['answer_start']
                    answer_end = answer_start + len(answer)
            else:  
                
                answer = ""
                answer_start = ""
                answer_end = ""
            test_df_list.append((context, question, is_impossible, answer, answer_start, answer_end))
test_df = pd.DataFrame(test_df_list, columns = ['Context', 'Question', 'Is_possible','Answer', 'Answer_Start', 'Answer_End'])


# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open('C:\\Saurabh\\Adv NLP\\machine_learning_examples\\glove.6B.50d.txt',  encoding="utf8") as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))





train_df = test_df.copy()

# some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 100000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 256
EPOCHS = 10
LATENT_DIM = 25



tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer.fit_on_texts(pd.concat([train_df.Context, train_df.Question]))
context_sequences = tokenizer.texts_to_sequences(train_df.Context.values)
question_sequences = tokenizer.texts_to_sequences(train_df.Question.values)
answer_sequences = tokenizer.texts_to_sequences(train_df.Answer.values)

inverse_word_index= {value: key for key, value in tokenizer.word_index.items()}

max_sequence_length_from_data = max(len(s) for s in context_sequences)
print('Max sequence length:', max_sequence_length_from_data)
#Max sequence length: 653)

max_sequence_length_from_data = max(len(s) for s in question_sequences)
print('Max sequence length:', max_sequence_length_from_data)
#Max sequence length: 38


max_sequence_length_from_data = max(len(s) for s in answer_sequences)
print('Max sequence length:', max_sequence_length_from_data)
#Max sequence length: 38

word2idx = tokenizer.word_index
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)


# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector


context_sequences = pad_sequences(context_sequences, maxlen=700, padding='pre')
question_sequences = pad_sequences(question_sequences, maxlen=40, padding='post')
answer_sequences = pad_sequences(answer_sequences, maxlen=20, padding='post')



#one_hot_targets = np.zeros((len(answer_sequences), 20, num_words))
#for i, target_sequence in enumerate(answer_sequences):
#  for t, word in enumerate(target_sequence):
#    if word > 0:
#      one_hot_targets[i, t, word] = 1

one_hot_targets = np.zeros((len(answer_sequences), num_words))
for i, target_sequence in enumerate(answer_sequences):
  for t, word in enumerate(target_sequence):
    if word > 0:
      one_hot_targets[i, t] = 1


hidden_size = 100

sentence_passage = Input(shape=(700,), dtype='int32')
#encoded_sentence_passage = Embedding(len(word2idx)+1, hidden_size)(context_sequences)
encoded_sentence_passage = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], trainable = False)(sentence_passage)
encoded_sentence_passage = Bidirectional(LSTM(hidden_size))(encoded_sentence_passage)

question = Input(shape=(40,), dtype='int32')
encoded_question = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], trainable = False)(question)
encoded_question = Bidirectional(LSTM(hidden_size))(encoded_question)
#encoded_question = RepeatVector(700)(encoded_question)

merge = layers.concatenate([encoded_sentence_passage, encoded_question])
#merge = Bidirectional(LSTM(hidden_size))(merge)
#predicted = Dense(len(inverse_word_index)+1, activation='softmax')(merge)
predicted = Dense(len(inverse_word_index)+1, activation='softmax')(merge)

model = Model([sentence_passage, question], predicted)
model.compile(optimizer=keras.optimizers.Adam(lr = 0.003), loss='sparse_categorical_crossentropy' )
model.fit( [context_sequences, question_sequences], answer_sequences, batch_size=128, epochs=EPOCHS,
          validation_split = 0.2)



input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# equivalent to added = keras.layers.add([x1, x2])
added = keras.layers.Concatenate()([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)



























