import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model 
import os, sys
import numpy as np
from random import shuffle
import argparse
import math
from math import log, floor
import pandas as pd
import csv


def load_data(train_data, train_label, test_data):
    x_train = pd.read_csv(train_data, sep=',', header=0)
    x_train = np.array(x_train.values)

    y_train = pd.read_csv(train_label, sep=',', header=0)
    y_train = np.array(y_train.values)


    x_test = pd.read_csv(test_data, sep=',', header=0)
    x_test = np.array(x_test.values)

    return (x_train, y_train, x_test)

def normalize(x_all,x_test):
    x_all_test = np.concatenate((x_all,x_test))
    mu = (sum(x_all_test))/(x_all_test.shape[0])
    sigma = np.std(x_all_test,axis=0)

    mu = np.tile(mu,(x_all_test.shape[0],1))
    sigma = np.tile(sigma, (x_all_test.shape[0], 1))
    x_all_test_normed = (x_all_test-mu)/sigma

    x_all = x_all_test_normed[0:x_all.shape[0]]
    x_test = x_all_test_normed[x_all.shape[0]:]

    return x_all,x_test



##
x_train, y_train, x_test = load_data(sys.argv[1],sys.argv[2],sys.argv[3])
x_train, x_test = normalize(x_train,x_test)
y_train = np_utils.to_categorical(y_train)
#print(y_train)

"""
##model
model = Sequential()
model.add(Dense(input_dim = 106,units = 54,activation='sigmoid'))
#model.add(Dense(units = 26,activation='sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(units = 2,activation='sigmoid'))
#model.add(Dropout(0.5))
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size = 32,epochs = 30)

##before load
result = model.predict(x_test)
##save
model.save('my_model.h5')

##print prediction
ans = []
for i in range(len(result)):
    ans.append([str(i+1)])
    re = result[i]
    if re[0]>re[1]:
        a = 0
    elif re[0]<=re[1]:
        a = 1    
    ans[i].append(a)


filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()


del model
"""
##after load

model = load_model('my_model.h5')

re_result = model.predict(x_test)

ans = []
for i in range(len(re_result)):
    ans.append([str(i+1)])
    re = re_result[i]
    if re[0]>re[1]:
        a = 0
    elif re[0]<=re[1]:
        a = 1    
    ans[i].append(a)


filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
