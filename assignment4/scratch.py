import keras
import tensorflow
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # This is now fully integrated into tensorflow
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model, Sequential
from keras import layers

import numpy as np
import pandas as pd

train = pd.read_csv('liar_dataset/train.tsv', sep='\t')
train.columns = ['ID', 'label', 'statement', 'subject', 'speaker', 'spkrJobTitle', 'state', 'party', 'barelyTrueCounts', 'falseCounts', 'halfTrueCounts', 'mostlyTrueCounts', 'pantsOnFireCounts', 'context']

test = pd.read_csv('liar_dataset/test.tsv', sep='\t')
test.columns = ['ID', 'label', 'statement', 'subject', 'speaker', 'spkrJobTitle', 'state', 'party', 'barelyTrueCounts', 'falseCounts', 'halfTrueCounts', 'mostlyTrueCounts', 'pantsOnFireCounts', 'context']


MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

numData = 50
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train['statement'].head(numData)) # DON'T FORGET TO RESET
train_sequences = tokenizer.texts_to_sequences(train['statement'].head(numData))  # DON"T FORGET TO RESET
test_sequences = tokenizer.texts_to_sequences(test['statement'].head(numData)) # DON'T FORGET TO RESET
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#Converting this to sequences to be fed into neural network. Max seq. len
# is 1000 as set earlier. Initial padding of 0s, until vector is of
#size MAX_SEQUENCE_LENGTH

# Make Data
train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Make Keras Labels
NUM_CLASSES = 6
y_train_enc, y_test_enc = prepare_targets(train['label'].head(numData), test['label'].head(numData)) # DON'T FORGET TO RESET
y_train = to_categorical(y_train_enc, NUM_CLASSES)
y_test = to_categorical(y_test_enc, NUM_CLASSES)


embeddings_index = {}
with open("C:\\Users\\jason\\Desktop\\glove.6B.100d.txt", encoding="utf8") as f: # DON'T FORGET TO CHANGE BACK TO RELATIVE PATH
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), input_length=MAX_SEQUENCE_LENGTH, trainable=False)
print("Preparing of embedding matrix is done")

myLSTM = Sequential()
myLSTM.add(Embedding(MAX_NUM_WORDS, 128))
myLSTM.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
myLSTM.add(Dense(6, activation='sigmoid'))
myLSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
result = myLSTM.fit(train_data, y_train, batch_size=32, epochs=1, validation_data=(test_data, y_test))
score, acc = myLSTM.evaluate(test_data, y_test, batch_size=32)
print('LSTM predication accuracy:', acc)
