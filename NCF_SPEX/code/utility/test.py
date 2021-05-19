import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data

# dataset name
dataset = 'epinion2t'
# assert dataset in ['ml-1m', 'pinterest-20']

# model name
model = 'NeuMF-end'
# assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = '../../Data/'

train_rating = main_path + '{}.train.rating'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = '../models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'

train_data = pd.read_csv(train_rating, sep=' ', header=None, names=['user', 'item'], usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
m = max(train_data['user'])
a = set(train_data['user'])
print(type(a))
print(m)