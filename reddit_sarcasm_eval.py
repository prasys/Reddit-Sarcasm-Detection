from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import numpy as np
import pandas as pd

def comments_to_seq2(c):
    seq = [[np.asarray(
        [0 if "..." in x else 1,
        0 if x.upper() == x else 1,
        0 if "!" in x else 1])
        for x in comment.split(" ")] for comment in c]
    return seq

MAX_NB_WORDS = 50000 #50000 unique words?
MAX_SEQUENCE_LENGTH = 30 #30 word sentences?

key_df =pd.read_csv('data/key.csv', sep='\t')
train_df = pd.read_csv('data/train-balanced.csv', sep='\t', header=None, names=list(key_df))
test_df = pd.read_csv('data/test-balanced.csv', sep='\t', header=None, names=list(key_df))
print("Read in CSVs")
comments = [str(x) for x in list(train_df["comment"])] #add in parent comment later
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(comments)
print("Fitted Tokenizer to text")

input_comments = [str(x) for x in list(test_df["comment"])] 

sequences = tokenizer.texts_to_sequences(input_comments)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
data2 = pad_sequences(comments_to_seq2(input_comments), maxlen=MAX_SEQUENCE_LENGTH)

model = load_model("large_feature_model_2.h5")
labels = to_categorical(np.asarray(test_df["label"]))
print(model.evaluate(x=[data, data2], y=labels))
print(model.metrics_names)
