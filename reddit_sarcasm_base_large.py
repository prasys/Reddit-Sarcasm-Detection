
# coding: utf-8

# In[54]:

import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Embedding, Concatenate, Merge, Input, concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

key_df =pd.read_csv('data/key.csv', sep='\t')

train_df = pd.read_csv('data/train-balanced.csv', sep='\t', header=None, names=list(key_df))
test_df = pd.read_csv('data/test-balanced.csv', sep='\t', header=None, names=list(key_df))

MAX_NB_WORDS = 50000 #50000 unique words?
MAX_SEQUENCE_LENGTH = 30 #30 word sentences?
comments = [str(x) for x in list(train_df["comment"])] #add in parent comment later
test_comments = [str(x) for x in list(test_df["comment"])] #add in parent comment later
#https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(train_df["label"]))
labels_test = to_categorical(np.asarray(test_df["label"]))

test_sequences = tokenizer.texts_to_sequences(test_comments)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

x_train = data
x_val = test_data
y_train = labels
y_val = labels_test

#https://keras.io/getting-started/sequential-model-guide/
EMBEDDING_DIM = 16
data_dim = EMBEDDING_DIM
timesteps = MAX_SEQUENCE_LENGTH
NUM_CLASSES = 2
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
embed_input = Input(shape=(30,))
first_embed = Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(embed_input)
first_lstm = LSTM(32, batch_input_shape=(batch_size, timesteps, data_dim))(first_embed)
first_dense = Dense(NUM_CLASSES, activation='softmax')(first_lstm)

model = Model(inputs=[embed_input], outputs=first_dense)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          validation_data=(x_val, y_val))

model.save("large_base_model_2.h5")



