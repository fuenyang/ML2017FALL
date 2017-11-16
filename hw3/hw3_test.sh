#!/bin/bash
wget 'https://www.dropbox.com/s/121vgss4dlspme8/model_49.h5?dl=1' -O 'model.h5'
python3 predict.py $1 $2