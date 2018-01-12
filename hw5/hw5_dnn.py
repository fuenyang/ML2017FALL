import sys
import numpy as np
import keras
from keras.layers import Embedding, Flatten, Dot, Input, Concatenate, Dense
from keras.models import Sequential, Model
from keras.models import load_model

def load_data(fname):
    users = []
    movies = []
    ratings = []
    with open(fname) as f:
        next(f)
        for line in f:
            tid, user, movie, rating = line.strip().split(',')
            users.append(user)
            movies.append(movie)
            ratings.append(rating)
    users = np.array(users, dtype=np.int)
    movies = np.array(movies, dtype=np.int)
    ratings = np.array(ratings, dtype=np.float)
    return users, movies, ratings

def loadUsersData(fname):
    with open(fname,'r') as f:
        next(f)
        Age = [0]*899873
        
        for row in f:
            userid, gender, age, occu, zipcode = row.strip().split('::')
            Age[int(userid)] = age
    Age = np.array(Age, dtype=np.float)
    return Age


users, movies, ratings = load_data('train.csv')
ages = loadUsersData('users.csv')
Ages = []
for i in users:
    Ages.append(ages[i])
Ages = np.array(Ages,dtype=np.float)
print(Ages)
print(Ages.shape)

n_users = 6041
n_items = 3953

dim = 75

input_user = Input(shape=[1])
input_item = Input(shape=[1])
input_age = Input(shape=[1])

emb_user = Embedding(n_users, dim, embeddings_initializer='random_normal')(input_user)
emb_user = Flatten()(emb_user)
emb_item = Embedding(n_items, dim, embeddings_initializer='random_normal')(input_item)
emb_item = Flatten()(emb_item)
emb_age = Embedding(n_users, dim, embeddings_initializer='random_normal')(input_age)
emb_age = Flatten()(emb_age)


merge = Concatenate()([emb_user,emb_item,emb_age])
hidden = Dense(150,activation='relu')(merge)
hidden = Dense(50,activation='relu')(hidden)

output = Dense(1)(hidden)
model = Model([input_user, input_item, input_age], output)
model.compile(loss='mean_squared_error', optimizer='adamax')
model.fit([users, movies, Ages], ratings, batch_size=256 ,epochs=18)
model.save('model_mf_dnn.h5')

# def load_testdata(fname):
#     users = []
#     movies = []
#     with open(fname) as f:
#         next(f)
#         for line in f:
#             tid, user, movie = line.strip().split(',')
#             users.append(user)
#             movies.append(movie)
#     users = np.array(users, dtype=np.int)
#     movies = np.array(movies, dtype=np.int)
#     return users, movies

# users, movies = load_testdata('test.csv')

# # model = load_model('model_mf_nor.h5')
# ans = model.predict([users, movies, Ages])

# with open('prediction_mf_dnn.csv', 'w') as f:
#     f.write('TestDataID,Rating\n')
#     for ids, ratings in enumerate(ans):
#         f.write('{},{}\n'.format(ids + 1, ratings[0]))
