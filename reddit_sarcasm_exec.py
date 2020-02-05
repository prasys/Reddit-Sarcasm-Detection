from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd

def comments_to_seq2(c):
    seq = [[np.asarray(
        [0 if "..." in x else 1,
        0 if x.upper() == x else 1,
        0 if "!" in x else 1])
        for x in comment.split(" ")] for comment in c]
    return seq

MAX_NB_WORDS = 50000 #2000 unique words?
MAX_SEQUENCE_LENGTH = 30 #30 word sentences?

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
#input_comments = ["yes sir yes", "I  love fridays","Yay that was so much fun when people would go up there with the flag and wait for people to quit so they could get a win."]
df = pd.read_csv('file_name.csv')
input_comments = df['Comment'].to_list()
sequences = tokenizer.texts_to_sequences(input_comments)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
data2 = pad_sequences(comments_to_seq2(input_comments), maxlen=MAX_SEQUENCE_LENGTH)
model = load_model("large_feature_model_2.h5")
print(model.predict([data, data2]))
