import numpy as np

import csv
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, LSTM, Flatten
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
from gensim.models import word2vec
import pickle

models = word2vec.Word2Vec.load('modwtov_100.model')

def loadTestData(test_filename):
    file = open(test_filename, 'r')
    data = []
    row_num = 0
    for row in file.readlines():
        if row_num != 0:
            temp = row.find(',')
            temp = temp + 1
            row = row[temp:len(row)]
            data.append(row)
            
        row_num = row_num + 1
    return data

data_test = loadTestData(sys.argv[1])

with open('tokenizer.pkl', 'rb') as handle:
    t = pickle.load(handle)

word_index = t.word_index

data_test_seq  = t.texts_to_sequences(data_test)
padded_data_test = pad_sequences(data_test_seq,maxlen =60)

vocab_size = len(t.word_index) + 1
batch_size = 256
inshape = 100
indim = 1


model = load_model('model.h5')
result = model.predict(padded_data_test)

# produce output file

ans = []
for i in range(len(result)):
    ans.append([str(i)])
    re = result[i]
    if re<=0.5:
        a = 0
    elif re>0.5:
        a = 1
    

    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()



