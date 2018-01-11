import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import sklearn
import csv
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
import pickle
import sys

def loadTestData(filename):
    imginx_1 = []
    imginx_2 = []
    with open(filename,'r') as f:
        next(f)
        for row in f:
            ID, im1, im2 = row.strip().split(',')
            im1 = int(im1)
            im2 = int(im2)
            imginx_1.append(im1)
            imginx_2.append(im2)
    return imginx_1, imginx_2




image = np.load(sys.argv[1])

image = image.astype('float32') / 255.

# #PCA, TruncatedSVD
# image_emb = PCA(n_components=50).fit_transform(image)
# print(image_emb.shape)

# #Auto-encoder
# encoding_dim = 32

# input_img = Input(shape=(784,))
# encoded = Dense(256, activation='relu')(input_img)
# encoded = Dense(128, activation='relu')(encoded)
# encoded = Dense(64, activation='relu')(encoded)
# encoded = Dense(32, activation='relu')(encoded)

# decoded = Dense(64, activation='relu')(encoded)
# decoded = Dense(128, activation='relu')(decoded)
# decoded = Dense(256, activation='relu')(decoded)
# decoded = Dense(784, activation='sigmoid')(decoded)

# autoencoder = Model(input_img, decoded)
# encoder = Model(input_img, encoded)







# autoencoder.compile(optimizer='adamax', loss='mse')
# autoencoder.fit(image, image,
#                 epochs=1500,
#                 batch_size=256,
                
#                 )



# encoder.save('encoder_3.h5')

encoder = load_model('encoder_3.h5')
print('encoder loaded successfully!')
#clustering
image_emb = encoder.predict(image)
image_clu = KMeans(n_clusters=2, random_state=0).fit(image_emb)
# with open('image_kmeans.pickle_3', 'wb') as f:
#     pickle.dump(image_clu, f)




#load testing data
image_inx1, image_inx2 = loadTestData(sys.argv[2])




#predict
ans = []
for i in range(len(image_inx1)):
    ans.append([str(i)])
    image_pre = image_clu.predict([image_emb[image_inx1[i]],image_emb[image_inx2[i]]])
    # print(image_pre)
    # print('im0:',image_pre[0])
    # print('im1:',image_pre[1])
    
    if image_pre[0]==image_pre[1]:
        a = 1
    elif image_pre[0]!=image_pre[1]:
        a = 0
    # print('a=',a)

    ans[i].append(a)
            



filename = sys.argv[3]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["ID","Ans"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()



