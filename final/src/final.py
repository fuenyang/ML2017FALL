from gensim.models import word2vec
import numpy as np
import csv
import sys
import os
import argparse
import jieba
from numpy import linalg as LA
import io
import math


stopwordset = set()
with open('stopwords.txt','r') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))


def chineseCut(filename):
    data = []
    with open(filename,'r') as content:
        for row in content:
            row = row.strip()
            word=jieba.cut(row, cut_all=False)
            word1=" ".join(word)
            data.append(word1)
        

    return data

def loadTestData(filename,stopwordset):
    with open(filename,'r') as f:
        next(f)
        question = []
        option1 = []
        option2 = []
        option3 = []
        option4 = []
        option5 = []
        option6 = []


        for row in f:
            row = row.strip()
            iden, qu, op = row.split(',')
            tmpqu = []
            for word in jieba.cut(qu, cut_all=False):
                if word not in stopwordset:
                    tmpqu.append(word)
              
            question.append(tmpqu)
            op1, op2, op3, op4, op5, op6 = op.split('\t')
            tmpop1 = []
            tmpop2 = []
            tmpop3 = []
            tmpop4 = []
            tmpop5 = []
            tmpop6 = []
            for word in jieba.cut(op1, cut_all=False):
                if word not in stopwordset:
                    tmpop1.append(word)
            for word in jieba.cut(op2, cut_all=False):
                if word not in stopwordset:
                    tmpop2.append(word)
            for word in jieba.cut(op3, cut_all=False):
                if word not in stopwordset:
                    tmpop3.append(word)
            for word in jieba.cut(op4, cut_all=False):
                if word not in stopwordset:
                    tmpop4.append(word)
            for word in jieba.cut(op5, cut_all=False):
                if word not in stopwordset:
                    tmpop5.append(word)
            for word in jieba.cut(op6, cut_all=False):
                if word not in stopwordset:
                    tmpop6.append(word)

            option1.append(tmpop1)
            option2.append(tmpop2)
            option3.append(tmpop3)
            option4.append(tmpop4)
            option5.append(tmpop5)
            option6.append(tmpop6)
        



    return question ,option1, option2, option3, option4, option5, option6

def construtVector(sen_list,models):
    
    question_vec = []
    for rows in sen_list:
        temp = np.zeros(300)
        for ele in rows:
            if ele in models.wv.vocab:
                temp = temp+np.array(models.wv[ele])
            else:
                temp = temp
            
            # try:
            #     temp = temp+model_2[ele]
            #     print(temp)
            #     break
            # except KeyError:
            #     temp = temp
                
                
        question_vec.append(temp.tolist())
    return question_vec
def computeSimilarity(vec1,vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    simi = np.dot(vec1,vec2)
    # simi = np.true_divide(simi,LA.norm(vec1)*LA.norm(vec2))
    return simi






# f = open('all_train.txt', 'w', encoding = 'utf-8')

# with open('1_train.txt','r',encoding='utf-8') as f1:
#     da1 = f1.read()
#     f.write(da1)
    

# with open('2_train.txt','r',encoding='utf-8') as f2:
#     da2 = f2.read()
#     f.write(da2)

# with open('3_train.txt','r',encoding='utf-8') as f3:
#     da3 = f3.read()
#     f.write(da3)

# with open('4_train.txt','r',encoding='utf-8') as f4:
#     da4 = f4.read()
#     f.write(da4)

# with open('5_train.txt','r',encoding='utf-8') as f5:
#     da5 = f5.read()
#     f.write(da5)

# f.close()


# main_datafile = 'all_train.txt'
# output_file = 'trainSeq.txt'
# output = io.open(output_file,'w',encoding='utf-8')

# with io.open(main_datafile, 'r', encoding='utf-8') as content:
#     for line in content:
#         words = jieba.cut(line, cut_all=False)
#         wordcount = 0
#         for id ,word in enumerate(words):
#             if word !='\n' and word!=' ':
#                 output.write(word+' ')
#                 wordcount = wordcount+1
#         if wordcount!=0:
#             output.write(u'\n')
# output.close()






# data_all = data_train1+data_train2+data_train3+data_train4+data_train5


word_dim = 300

# sentence = word2vec.Text8Corpus('trainSeq.txt')
# models = word2vec.Word2Vec(sentence, size=word_dim, min_count=5, workers=4, iter=25)
# models.save('modwtov_fi300_4.model')


question_test ,option1_test, option2_test, option3_test, option4_test, option5_test, option6_test= loadTestData(sys.argv[1],stopwordset)

models = word2vec.Word2Vec.load('modwtov_fi300_4_best.model')




qu_vec = construtVector(question_test,models)
op1_vec = construtVector(option1_test,models)
op2_vec = construtVector(option2_test,models)
op3_vec = construtVector(option3_test,models)
op4_vec = construtVector(option4_test,models)
op5_vec = construtVector(option5_test,models)
op6_vec = construtVector(option6_test,models)


si0 = 0
si1 = 0
si2 = 0
si3 = 0
si4 = 0
si5 = 0
ans = []


for i in range(len(qu_vec)):
    ans.append([str(i+1)])
    si0 = computeSimilarity(qu_vec[i],op1_vec[i])
    si1 = computeSimilarity(qu_vec[i],op2_vec[i])
    si2 = computeSimilarity(qu_vec[i],op3_vec[i])
    si3 = computeSimilarity(qu_vec[i],op4_vec[i])
    si4 = computeSimilarity(qu_vec[i],op5_vec[i])
    si5 = computeSimilarity(qu_vec[i],op6_vec[i])
    

    if max(si0,si1,si2,si3,si4,si5)==si0:
        a=0
        
    elif max(si0,si1,si2,si3,si4,si5)==si1: 
        a=1
        
    elif max(si0,si1,si2,si3,si4,si5)==si2: 
        a=2
        
    elif max(si0,si1,si2,si3,si4,si5)==si3: 
        a=3
        
    elif max(si0,si1,si2,si3,si4,si5)==si4: 
        a=4
        
    elif max(si0,si1,si2,si3,si4,si5)==si5: 
        a=5
        
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","ans"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()