import numpy as np 
import pandas as pd 
import scipy.sparse as sp
import torch.utils.data as data
import utility.config as config

def load_all(test_num=100):
	""" We load all the three file here to save time in each epoch. """
	train_data = pd.read_csv(config.train_rating, sep=' ', header=None, names=['user', 'item'], usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

	user_num = train_data['user'].max() + 1
	item_num = train_data['item'].max() + 1

	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0
	testRatings = load_test_rating_as_dict(config.test_rating)
	testNegatives = load_test_negative_as_dict(config.test_negative)
	
	return train_data, testRatings, testNegatives, user_num, item_num, train_mat

def load_test_rating_as_dict(filename):
    ratingdict = {}
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                arr = line.split(" ")  # \t
                user, item = int(arr[0]), int(arr[1])
                ratingdict[user] = [item]
    return ratingdict

def load_test_negative_as_dict(filename):
    negativedict = {}
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                arr = line.split(" ")
                user = int(arr[0])
                negatives = []
                for x in arr[1: ]:
                    x = x.strip()
                    if x == "":
                        continue
                    negatives.append(int(x))
                negativedict[user] = negatives
    return negativedict

class NCFData(data.Dataset):
	def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
		super(NCFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features_ps = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
		self.labels = [0 for _ in range(len(features))]

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'

		self.features_ng = []
		for x in self.features_ps:
			u = x[0]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])

		labels_ps = [1 for _ in range(len(self.features_ps))]
		labels_ng = [0 for _ in range(len(self.features_ng))]

		self.features_fill = self.features_ps + self.features_ng
		self.labels_fill = labels_ps + labels_ng

	def __len__(self):
		return (self.num_ng + 1) * len(self.labels)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training else self.features_ps
		labels = self.labels_fill if self.is_training else self.labels

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		return user, item ,label
		