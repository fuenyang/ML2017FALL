import sys
import numpy as np
import keras
from keras.layers import Embedding, Flatten, Dot, Input, Add
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

np.random.seed(0)

users, movies, ratings = load_data('train.csv')

# #normalize
# ratings_nor = (ratings-ratings.mean())/ratings.std()

n_users = 6041
n_items = 3953
dim = 110

input_user = Input(shape=[1])
input_item = Input(shape=[1])
emb_user = Embedding(n_users, dim)(input_user)
emb_user = Flatten()(emb_user)
emb_item = Embedding(n_items, dim)(input_item)
emb_item = Flatten()(emb_item)

# bias_user = Embedding(n_users, 1, embeddings_initializer='zeros')(input_user)
# bias_user = Flatten()(bias_user)
# bias_item = Embedding(n_items, 1, embeddings_initializer='zeros')(input_item)
# bias_item = Flatten()(bias_item)

out = Dot(axes=1)([emb_user, emb_item])

# out = Add()([out, bias_user, bias_item])

model = Model([input_user, input_item], out)
model.compile(loss='mean_squared_error', optimizer='adamax')
model.fit([users, movies], ratings, batch_size=256 ,epochs=18)

#get movie embedding
user_emb = np.array(model.layers[2].get_weights()).squeeze()
print('user embedding shape:',user_emb.shape)
movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print('movie embedding shape:',movie_emb.shape)
np.save('user_emb.npy', user_emb)
np.save('movie_emb.npy', movie_emb)

model.save('model_mf_110.h5')

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
# ans = model.predict([users, movies])

# with open('prediction_mf_110.csv', 'w') as f:
#     f.write('TestDataID,Rating\n')
#     for ids, ratings in enumerate(ans):
#         f.write('{},{}\n'.format(ids + 1, ratings[0]))