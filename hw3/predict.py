import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.models import load_model
import sys


data_test = []
with open(sys.argv[1]) as img:
    next(img)

    for row in img:
        idx, data = row.strip().split(',')
        data_test.append(data.split(' '))

data_test = np.array(data_test,dtype=np.float)
data_test = (data_test-data_test.mean())/data_test.std()

data_test = data_test.reshape(data_test.shape[0],48,48,1)

model = load_model('model.h5')

result = model.predict(data_test)



# produce output file

ans = []
for i in range(len(result)):
    ans.append([str(i)])
    re = result[i]
    if max(re)==re[0]:
        a = 0
    elif max(re)==re[1]:
        a = 1
    elif max(re)==re[2]:
        a = 2
    elif max(re)==re[3]:
        a = 3
    elif max(re)==re[4]:
        a = 4
    elif max(re)==re[5]:
        a = 5
    elif max(re)==re[6]:
        a = 6

    ans[i].append(a)


filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()




