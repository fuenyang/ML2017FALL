import numpy as np
from numpy.linalg import inv
import random
import math
import csv
import sys


# read model
w = np.load('modelhw1_best.npy')


test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)




#get ans.csvwith my model

test1 = []
for i in range(240):
    test1.append([])
temp = []
for i in range(len(test_x)):
    temp = test_x[i].tolist()
    test1[i].append(temp[0])
    for j in range(0,9):
        test1[i].append(temp[82+j])
    for k in range(0,9):
        test1[i].append(temp[163+81+k])        
#print(test1)        
  


ans = []
for i in range(len(test1)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test1[i])
    ans[i].append(a)




filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()