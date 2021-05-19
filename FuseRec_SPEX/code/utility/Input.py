import numpy as np
from torch.utils.data import TensorDataset
import pickle
import random
import torch


class Dataset():
    def __init__(self, dataset, root="../data2/"):
        # ---- load train file ---------
        self.dataset = dataset
        self.root = root
        train_file = root + dataset + "/rec/train.txt"
        xy = np.loadtxt(train_file, delimiter=',', dtype=np.int64)
        self.x_data = xy[:, 0]
        self.y_data = xy[:, 1]
        self.t_data = xy[:, 2]
        self.train_len = xy.shape[0]
        self.user_num = max(self.x_data) + 1
        self.item_num = max(self.y_data) + 1

        # ---- load train file ---------
        test_file = root + dataset + "/rec/test.txt"
        xyt = np.loadtxt(test_file, delimiter=',', dtype=np.int64)
        self.xt_data = xyt[:, 0]
        self.yt_data = xyt[:, 1]
        self.test_len = xyt.shape[0]

        # ---- load item type file ---------
        relation_file = root + dataset + "/rec/relation.pickle"
        self.u_items_dic, self.u_frids_dic, self.F_i_dic = pickle.load(open(relation_file, 'rb'))

    def generate_train_data(self):
        user_all, item_all, label_all, u_items_all, u_items_mask_all, u_frids_all, u_frids_mask_all, u_frids_items_all, F_i_all = [], [], [], [], [], [], [], [], []
        for index in range(self.train_len):
            # ---- pos ---------
            user = self.x_data[index]
            item = self.y_data[index]

            item_t_ind = self.t_data[index] - 1
            if item_t_ind == 0: continue
            if item_t_ind <= 30:
                u_items = self.u_items_dic[user][:item_t_ind] + [self.item_num] * (30 - item_t_ind)
                u_items_mask = item_t_ind
            else:
                u_items = self.u_items_dic[user][item_t_ind - 30:item_t_ind]
                u_items_mask = 30

            u_frids = self.u_frids_dic[user]
            u_frids_l = len(u_frids)
            if u_frids_l < 10:
                u_frids.extend([self.user_num] * (10 - u_frids_l))
                u_frids_mask = u_frids_l
            else:
                u_frids = random.sample(u_frids, 10)
                u_frids_mask = 10

            u_frids_items = []
            for f in u_frids:
                ufis = self.u_items_dic[f]
                if f == self.user_num:
                    u_frids_items.append([0] * 3)
                else:
                    u_frids_items.append(random.sample(ufis, 3))

            pos_F_i = self.F_i_dic[item] 

            # ---- neg sample---------
            neg_items = random.sample(set(range(self.item_num)).difference(set(self.u_items_dic[user])), 5)
            neg_F_i = [self.F_i_dic[ni] for ni in neg_items]

            # ------------------------
            user_all.extend([user] * 6)
            item_all.extend([item] + neg_items)
            label_all.extend([1.] + [0.] * 5)
            u_items_all.extend([u_items] * 6)
            u_items_mask_all.extend([u_items_mask] * 6)
            u_frids_all.extend([u_frids] * 6)
            u_frids_mask_all.extend([u_frids_mask] * 6)
            u_frids_items_all.extend([u_frids_items] * 6)
            F_i_all.extend(neg_F_i + [pos_F_i])

        trainset = TensorDataset(torch.LongTensor(user_all), torch.LongTensor(item_all), torch.FloatTensor(label_all),
                                 torch.LongTensor(u_items_all), torch.LongTensor(u_items_mask_all),
                                 torch.LongTensor(u_frids_all),
                                 torch.LongTensor(u_frids_mask_all), torch.LongTensor(u_frids_items_all),
                                 torch.LongTensor(F_i_all))

        return trainset


    def generate_test_data(self):
        user_all, item_all, label_all, u_items_all, u_items_mask_all, u_frids_all, u_frids_mask_all, u_frids_items_all, F_i_all = [], [], [], [], [], [], [], [], []
        for index in range(0, self.test_len, 100):
            user = self.xt_data[index]
            item = self.yt_data[index + 99]

            u_i_l = len(self.u_items_dic[user])
            if u_i_l <= 30:
                u_items = self.u_items_dic[user] + [self.item_num] * (30 - u_i_l)
                u_items_mask = u_i_l
            else:
                u_items = self.u_items_dic[user][u_i_l - 30:u_i_l]
                u_items_mask = 30

            u_frids = self.u_frids_dic[user]
            u_frids_l = len(u_frids)
            if u_frids_l < 10:
                u_frids.extend([self.user_num] * (10 - u_frids_l))
                u_frids_mask = u_frids_l
            else:
                u_frids = random.sample(u_frids, 10)
                u_frids_mask = 10

            u_frids_items = []
            for f in u_frids:
                ufis = self.u_items_dic[f]
                if f == self.user_num:
                    u_frids_items.append([0] * 3)
                else:
                    u_frids_items.append(random.sample(ufis, 3))

            pos_F_i = self.F_i_dic[item]

            # ---- neg sample---------
            neg_items = self.yt_data[index:index + 99]
            neg_F_i = [self.F_i_dic[ni] for ni in neg_items]

            # ------------------------
            user_all.extend([user] * 100)
            item_all.extend(np.append(neg_items, item))
            label_all.extend([0.] * 99 + [1.])
            u_items_all.extend([u_items] * 100)
            u_items_mask_all.extend([u_items_mask] * 100)
            u_frids_all.extend([u_frids] * 100)
            u_frids_mask_all.extend([u_frids_mask] * 100)
            u_frids_items_all.extend([u_frids_items] * 100)
            F_i_all.extend(neg_F_i + [pos_F_i])

        testset = TensorDataset(torch.LongTensor(user_all), torch.LongTensor(item_all), torch.FloatTensor(label_all),
                                torch.LongTensor(u_items_all), torch.LongTensor(u_items_mask_all),
                                torch.LongTensor(u_frids_all),
                                torch.LongTensor(u_frids_mask_all), torch.LongTensor(u_frids_items_all),
                                torch.LongTensor(F_i_all))

        return testset

    def load_type(self):
        file = self.root + self.dataset + "/rec/type.txt"
        yc = np.loadtxt(file, delimiter=' ', dtype=np.int)  
        c_data = yc[:, 1]
        return c_data
