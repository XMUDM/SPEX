import torch.utils.data as data
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError

    # @property
    # def testRatings(self):
    #     raise NotImplementedError
    #
    # @property
    # def testNegatives(self):
    #     raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config):
        dataset = config.dataset
        path = "../data/" + '{}/'.format(dataset)
        # train or test
        self.split = config.A_split
        self.folds = config.a_fold
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + 'rec/{}.train.rating'.format(dataset)
        test_rating_file = path + 'rec/{}.test.rating'.format(dataset)
        test_negative_file = path + 'rec/{}.test.negative'.format(dataset)
        self.path = path
        self.traindataSize = 0
        self.testDataSize = 0


        train_data = pd.read_csv(train_file, sep=' ', header=None, names=['user', 'item'], usecols=[0, 1],dtype={0: np.int32, 1: np.int32})
        self.trainUser = np.array(train_data['user'])
        self.trainItem = np.array(train_data['item'])
        self.n_user = max(train_data['user']) + 1
        self.m_item = max(train_data['item']) + 1
        self.trainUniqueUsers = np.array(list(set(train_data['user'])))

        self.rec_train_data = train_data.values.tolist()
        self.train_mat = sp.dok_matrix((self.n_user+1, self.m_item), dtype=np.float32)
        for x in self.rec_train_data:
            self.train_mat[x[0], x[1]] = 1.0

        self.testRatings = self.load_test_rating_as_dict(test_rating_file)
        self.testNegatives = self.load_test_negative_as_dict(test_negative_file)
        
        self.Graph = None
        print(dataset)
        print("use:", self.n_user)
        print("item:", self.m_item)
        print("----------------")
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user+1, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos



    def load_test_rating_as_dict(self, filename):
        ratingdict = {}
        with open(filename, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line != "":
                    arr = line.split(" ")  # \t
                    user, item = int(arr[0]), int(arr[1])
                    ratingdict[user] = [item]
        return ratingdict

    def load_test_negative_as_dict(self,filename):
        negativedict = {}
        with open(filename, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line != "":
                    arr = line.split(" ")
                    user = int(arr[0])
                    negatives = []
                    for x in arr[1:]:
                        x = x.strip()
                        if x == "":
                            continue
                        negatives.append(int(x))
                    negativedict[user] = negatives
        return negativedict

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users+1 + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users+1 + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().cuda())
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        # print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + 's_pre_adj_mat.npz')
                # print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()  #  self.n_users + self.m_items = 25100
                adj_mat = sp.dok_matrix((self.n_users+1 + self.m_items, self.n_users+1 + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users+1, self.n_users+1:] = R
                adj_mat[self.n_users+1:, :self.n_users+1] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                # print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + 's_pre_adj_mat.npz', norm_adj)

            if self.split == True:  # False
                self.Graph = self._split_A_hat(norm_adj)
                # print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().cuda()
                print("self.Graph:",self.Graph.size())
                # print("don't split the matrix")
        return self.Graph


    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))


class LightTrainData(data.Dataset):
    def __init__(self, features, num_item, train_mat=None):
        super(LightTrainData, self).__init__()
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = 5
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):

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
        features = self.features_fill
        labels = self.labels_fill

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label
