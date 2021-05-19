import pickle
import numpy as np
import tensorflow as tf
from collections import defaultdict

class Data():
    def __init__(self, data, n_node, shuffle=False, graph=None, test=False):
        self.test = test
        inputs = data[0]
        self.n_node = n_node # for padding
        self.len_max = 5
        inputs, mask = self.data_masks(inputs, [self.n_node]) # padding
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.targets = np.asarray(data[1])
        if test:
            self.neg = np.asarray(data[2])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            if self.test:
                self.neg = self.neg[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs = tf.gather(tf.convert_to_tensor(self.inputs), i)
        mask = tf.gather(tf.convert_to_tensor(self.mask), i)
        targets = tf.gather(tf.convert_to_tensor(self.targets), i)

        if self.test:
            negs = tf.gather(tf.convert_to_tensor(self.neg), i)
            return inputs, mask, targets, negs
        else:
            return inputs, mask, targets


    def data_masks(self, all_usr_pois, item_tail):  # inputs, [self.n_node]
        us_pois = []
        us_msks = []
        i = 0
        for upois in all_usr_pois:
            us_pois.append(upois + item_tail * (self.len_max - len(upois)))
            us_msks.append([1] * len(upois) + [0] * (self.len_max - len(upois)))
            i += 1
        return us_pois, us_msks  # inputs, mask, len_max


def loadData(args):
    trust_train_data = pickle.load(open(args.data_root + args.dataset + '/trust/train.txt', 'rb'))
    trust_test_data = pickle.load(open(args.data_root + args.dataset + '/trust/test2.txt', 'rb'))
    train_data = Data(trust_train_data, args.num_users, shuffle=True)
    test_data = Data(trust_test_data, args.num_users, shuffle=True, test=True)

    user_path_indx = defaultdict(list)
    path = trust_train_data[0]
    alluser_path_index = set(range(len(path)))
    for i, p in zip(range(len(path)), path):
        u = p[0]
        user_path_indx[u].append(i)

    return train_data,test_data,user_path_indx, alluser_path_index