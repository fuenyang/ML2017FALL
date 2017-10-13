import numpy as np
from numpy.linalg import inv
import random
import math
import csv
import sys

#read data
data = []

for i in range(18):
    data.append([])

n_row = 0
text = open('train.csv.csv','r',encoding='big5') #remember to change the inpute file path(MAYBE)
row = csv.reader(text,delimiter=',')
for r in row:
    if n_row!=0:
        for i in range(3,27):
            if r[i]!='NR':
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))
    n_row = n_row+1
text.close()

#parse data to (x,y)
x = []
y = []
for i in range(12):
    for j in range(480-9):
        x.append([])
        for t in range(9,10):
            for s in range(9): #can be change
                x[471*i+j].append(data[t][480*i+j+s] ) #can be change
        y.append(data[9][480*i+j+9])   #first '9' means the row of PM2.5,second '9' means PM2.5 of 10th hour #can be change
x = np.array(x)
y = np.array(y)

# add square term
x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

#print(len(x[0]))

#init weight & other hyperparams
w = np.zeros(len(x[0]))
l_rate = 100
repeat = 20000

#check your ans with close form solution
# use close form to check whether our gradient descent is good
# however, this cannot be used in hw1.sh 
w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)


#start training
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))
for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo-y
    cost = np.sum(loss**2)/len(x)
    cost_a = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w-l_rate*gra/ada
    print('iteration: %d | Cost: %f  ' % ( i,cost_a))

#print(w)     


# save model
np.save('modelhw1_best.npy',w)