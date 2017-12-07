#!/bin/bash
wget 'https://www.dropbox.com/s/zvnhs09k93qbw41/model_sp41.h5?dl=1' -O 'model.h5'
python3 rnn_predict.py $1 $2