import sys
import numpy as np
import keras
from keras.models import load_model

def load_testdata(fname):
    users = []
    movies = []
    with open(fname) as f:
        next(f)
        for line in f:
            tid, user, movie = line.strip().split(',')
            users.append(user)
            movies.append(movie)
    users = np.array(users, dtype=np.int)
    movies = np.array(movies, dtype=np.int)
    return users, movies

users, movies = load_testdata(sys.argv[1])

model = load_model('model_mf_110.h5')
ans = model.predict([users, movies])

with open(sys.argv[2], 'w') as f:
    f.write('TestDataID,Rating\n')
    for ids, ratings in enumerate(ans):
        f.write('{},{}\n'.format(ids + 1, ratings[0]))