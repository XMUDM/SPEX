import pandas as pd
import pickle
import torch
import numpy as np


def get_train_instances(u_train, i_train, tfset, n_items, n_users, max_friend):
    neg_num = 5
    input_user, input_item, label, input_uf = [], [], [], []

    pos = {}
    for i in range(len(u_train)):
        if u_train[i] in pos:
            pos[u_train[i]].append(i_train[i])
        else:
            pos[u_train[i]] = [i_train[i]]

    for i in range(len(u_train)):
        input_user.extend([u_train[i]] * (neg_num + 1))
        input_item.append(i_train[i])
        for i in range(neg_num):
            j = np.random.randint(n_items)
            while j in pos[u_train[i]]:
                j = np.random.randint(n_items)
            input_item.append(j)

        label.extend([1] + [0] * neg_num)

        one_use = n_users * np.ones(max_friend, np.int)
        if u_train[i] in tfset:
            one_use = tfset[u_train[i]]
        for i in range(neg_num + 1):
            input_uf.append(one_use)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(input_user), torch.LongTensor(input_item),torch.FloatTensor(label), torch.LongTensor(input_uf))

    return trainset


class Data(object):
    def __init__(self,path):

        train_file = path+'train.txt'
        test_file = path+'test.txt'
        trust_file = path+'friends.dic'
        neg_file = path+'negative.txt'

        self.tp_train = pd.read_csv(train_file, sep="\s+", names=['uid', 'sid'], usecols=[0, 1])
        self.tp_test = pd.read_csv(test_file, sep="\s+", names=['uid', 'sid'], usecols=[0, 1])

        self.n_users = max(self.tp_train['uid']) + 1
        self.n_items = max(self.tp_train['sid']) + 1

        self.tfset = pickle.load(open(trust_file, 'rb'), encoding='bytes')
        self.max_friend = len(self.tfset[list(self.tfset.keys())[0]])

        self.test_item, self.neg_item = {}, {}
        with open(test_file, 'r') as f:
            for line in f.readlines():
                l = line.strip().split(' ')
                self.test_item[int(l[0])] = [int(l[1])]

        with open(neg_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        line = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue
                    self.neg_item[line[0]] = line[1:]
