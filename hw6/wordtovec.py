from gensim.models import word2vec
import numpy as np
import jieba
from matplotlib import pyplot as plt 
from sklearn.manifold import TSNE
from adjustText import adjust_text
import pandas as pd

import matplotlib as mpl
font_name = "PMingLiU"                                                     
mpl.rcParams['font.sans-serif']=font_name 
mpl.rcParams['axes.unicode_minus']=False


def plot(Xs,Ys,Texts):
    plt.plot(Xs,Ys,'o')
    texts = [plt.text(X,Y,Text) for X,Y,Text in zip(Xs,Ys,Texts)]
    plt.title(str(adjust_text(texts,Xs,Ys,arrowprops=dict(arrowstyle='->',color='red'))))
    plt.show()



def chineseCut(filename):
    data = []
    with open(filename,'r') as content:
        for row in content:
            row = row.strip()
            tmp = []
            for word in jieba.cut(row, cut_all=False):
                tmp.append(word)
                
                
                    
            # print(tmp)
            
            data.append(tmp)
    return data

data = chineseCut('all_sents.txt')

model = word2vec.Word2Vec(data, size=120, min_count=20, workers=4)
model.save('modwtov_hw6_1.model')

model = word2vec.Word2Vec.load('modwtov_hw6_1.model')

vocabs = []                 
vecs = []                   
for vocab in model.wv.vocab:
    vocabs.append(vocab)
    vecs.append(model[vocab])
vecs = np.array(vecs)[:80]
vocabs = vocabs[:80]

tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vecs)
# red_x = reduced[:,0]
# red_y = reduced[:,1]

plt.figure()
texts = []
for i, label in enumerate(vocabs):
    
    x, y = reduced[i, :]
    texts.append(plt.text(x, y, label))
    plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

plt.savefig('hp.png', dpi=600)
plt.show()




