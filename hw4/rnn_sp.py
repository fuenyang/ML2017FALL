import numpy as np
import csv
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, LSTM, Flatten, Bidirectional, GRU
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
import sys
import os
import argparse
from gensim.models import word2vec
import math
from math import log, floor
import pickle



def loadTrainData(train_filename):
    file = open(train_filename, 'r')
    label = []
    data = []
    row_num = 0
    for row in file.readlines():
        elements = row.split(' ')
        label.append(elements[0])
        elements = elements[2:len(elements)]
        sentence = ' '.join(elements)
        data.append(sentence)
        row_num = row_num + 1
        
    label = np.array(label, dtype='int')
    return label, data
    
def loadTrainNoLabel(train_filename):
    file = open(train_filename, 'r')
    data = []
    for row in file.readlines():
        data.append(row)
    return data



def constructValidation(x, y, frac):
    
    length = x.shape[0]
    
    cv_idx_tmp = np.array([i for i in range(length)])
    cv_fold_1 = cv_idx_tmp[0:int(length*frac)]
    cv_fold_2 = cv_idx_tmp[int(length*frac):length]
    data_train = x[cv_fold_2[:], :]
    label_train = y[cv_fold_2[:]]
    data_val = x[cv_fold_1[:], :]
    label_val = y[cv_fold_1[:]]
    return data_train, label_train, data_val, label_val


        

label_train, data_train = loadTrainData(sys.argv[1])






# word embedding
with open('tokenizer.pkl', 'rb') as handle:
    t = pickle.load(handle)

word_index = t.word_index
data_train_seq = t.texts_to_sequences(data_train)



vocab_size = len(t.word_index) + 1
batch_size = 256
inshape = 100
indim = 1
epochsnum = 100
maxwordlen = 60

# label train

padded_data_train = pad_sequences(data_train_seq,maxlen =maxwordlen)






padded_data_train1, label_train, padded_data_train2, label_train2 = constructValidation(padded_data_train, label_train, 0.15)



models = word2vec.Word2Vec.load('modwtov_100.model')

embeddings_matrix = np.zeros([len(word_index)+1, 100], dtype='float32')

for keyword, index in word_index.items():
    if keyword in models.wv:
        embeddings_matrix[index] = models[keyword]







# build model
model = Sequential()
embedding_layer = Embedding(embeddings_matrix.shape[0],
                            embeddings_matrix.shape[1],
                            weights=[embeddings_matrix],
                            input_length=maxwordlen,
                            trainable=False) #trainable=False
model.add(embedding_layer)

# model.add(GRU(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))#, input_shape=(inshape,indim)
model.add(GRU(256, return_sequences=True, dropout=0.35, recurrent_dropout=0.35)) #, input_shape=(inshape,indim)
# model.add(GRU(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
# model.add(GRU(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
# model.add(GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
# model.add(LSTM(8, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
# model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
model.add(GRU(128, dropout=0.35, recurrent_dropout=0.35))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(1, activation='sigmoid'))


earlystopping = EarlyStopping(monitor='val_acc', patience = 5)
# compile $ fit
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(padded_data_train1, label_train,
          batch_size=batch_size,
          epochs=epochsnum,
          validation_data = (padded_data_train2, label_train2),
          callbacks=[earlystopping]) # validation_split=0.1



# save model
model.save('model.h5')

