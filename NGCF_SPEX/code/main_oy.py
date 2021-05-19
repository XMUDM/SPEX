import os
from ngcf_parser import parse_args
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

import math
import pickle
import random
from time import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
from utility.Logging import Logging
from utility.helper import *
from utility.batch_test import rec_test, data_generator, args
from utility2.trust_batch_test import trust_test5
from utility2.utils import Data
from utility2.layers import GraphAttentionLayer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(2020)

log_dir = os.path.join(os.getcwd(), 'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(os.getcwd(), 'log/%s_oy.log' % (args.dataset))
log = Logging(log_path)
log.record(args.act)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(2020)

class Model_Wrapper(nn.Module):
    def __init__(self, data_config, device):
        super(Model_Wrapper, self).__init__()
        self.device = device
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.embedding_dim = args.embed_size
        # trust
        self.all_path_index = config['all_path_index']
        self.trust_train_data = config['trust_train_data']
        self.trust_test_data = config['trust_test_data']
        self.user_path_indx = config['user_path_indx']
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.nonhybrid = args.nonhybrid
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        # share
        self.mlp = nn.Sequential()
        self.mlp.add_module("Linear_layer_s", nn.Linear(self.embedding_dim, self.embedding_dim))
        self.mlp.add_module("Relu_layer_s", nn.ReLU(inplace=True))

        self.mlp_r = nn.Sequential()
        self.mlp_r.add_module("Linear_layer_r", nn.Linear(self.embedding_dim * 2, self.embedding_dim))
        self.mlp_r.add_module("Relu_layer_r" , nn.ReLU(inplace=True))

        self.mlp_t = nn.Sequential()
        self.mlp_t.add_module("Linear_layer_t", nn.Linear(self.embedding_dim * 2, self.embedding_dim))
        self.mlp_t.add_module("Relu_layer_t" , nn.ReLU(inplace=True))

        self.reset_parameters()

        # multi head
        self.in_att = [GraphAttentionLayer(self.hidden_size, concat=True) for _ in range(args.nb_heads)]
        for i, attention in enumerate(self.in_att):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(self.hidden_size, concat=False)
        self.w = nn.Parameter(torch.zeros(size=(args.nb_heads * self.hidden_size, self.hidden_size)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)

        # rec
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.norm_adj = data_config['norm_adj']
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float()
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.mess_dropout = eval(args.mess_dropout)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.dropout_list.append(nn.Dropout(self.mess_dropout[i]))

        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # rec
        self.u2e_r = nn.Embedding(self.n_users+1, self.embedding_dim)
        nn.init.xavier_uniform_(self.u2e_r.weight)
        # trust
        self.u2e_t = nn.Embedding(self.n_users+1, self.embedding_dim)
        nn.init.xavier_uniform_(self.u2e_t.weight)


        # multi
        self.task_weights = nn.Parameter(torch.zeros((2), requires_grad=True))
        self.rec_loss_function = nn.BCEWithLogitsLoss()
        self.trust_loss_function = nn.CrossEntropyLoss()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, user, item, labels_list, trust_data, slice_indices, flag):
        share_u_r = self.mlp(self.u2e_r.weight)
        share_u_t = self.mlp(self.u2e_t.weight)
        self.u2e_r_new = self.mlp_r(torch.cat([self.u2e_r.weight, share_u_r], dim=1))[:self.n_users]
        self.u2e_t_new = self.mlp_t(torch.cat([self.u2e_t.weight, share_u_t], dim=1))

        if (flag == 0) or (flag == 1):
            ego_embeddings = torch.cat((self.u2e_r_new, self.item_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_layers):
                side_embeddings = torch.sparse.mm(self.norm_adj.to(self.device), ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)
                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.cat(all_embeddings, dim=1)
            ua_embeddings, ia_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            if (flag == 1): return ua_embeddings, ia_embeddings

            u_g_embeddings = ua_embeddings[trans_to_cuda(user)]
            i_g_embeddings = ia_embeddings[trans_to_cuda(item)]
            rec_loss = self.compute_rec_loss(u_g_embeddings, i_g_embeddings, labels_list)

        if (flag == 0) or (flag == 2):
            if flag == 0:
                inputs, mask, targets = trust_data.get_slice(slice_indices)
            else:
                inputs, mask, targets, negs = trust_data.get_slice(slice_indices)
                negs = trans_to_cuda(torch.Tensor(negs).long())

            inputs = trans_to_cuda(torch.Tensor(inputs).long())
            mask = trans_to_cuda(torch.Tensor(mask).long())
            targets = trans_to_cuda(torch.Tensor(targets).long())
            seq_l = torch.sum(mask, 1)
            # multi head
            mul_seq = trans_to_cuda(torch.cat([att(self.u2e_t_new, inputs, seq_l) for att in self.in_att], dim=2))
            mul_seq_c = torch.cat([mul_seq[i] for i in range(mul_seq.size()[0])], dim=0)
            mul_one = torch.mm(mul_seq_c, self.w)
            mul_one = F.elu(mul_one)
            seq_hidden = trans_to_cuda(self.out_att(self.u2e_t_new, mul_one.view(mul_seq.size()[0], mul_seq.size()[1], self.hidden_size), seq_l))
            trust_scores = self.compute_trust_score(seq_hidden, mask)

            if (flag == 2): return trust_scores, negs

            trust_loss = self.trust_loss_function(trust_scores, targets)

        return rec_loss, trust_loss


    def compute_rec_loss(self, u_g_embeddings, i_g_embeddings, labels_list):
        predict = torch.sum(torch.mul(u_g_embeddings, i_g_embeddings), dim=1)
        real_score = trans_to_cuda(labels_list)
        return self.rec_loss_function(predict, real_score)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_sparse_tensor_value(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

    # trust
    def compute_trust_score(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.u2e_t_new[:-1]
        trust_scores = torch.matmul(a, b.transpose(1, 0))
        return trust_scores

def train(model, optimizer):
    # rec
    best_recall, best_ndcg, best_iter = [0, 0, 0], [0, 0, 0], [0, 0]
    # trust2
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0]

    for epoch in range(args.epoch):
        t1 = time()
        loss, total_loss, total_loss1, total_loss2 = 0.0, 0.0, 0.0, 0.0
        # rec
        data_loader = data_generator.load_train_data()
        # trust
        avail_path_index = model.all_path_index
        trust_batch_size = len(avail_path_index) // len(data_loader)

        for data in data_loader:
            # rec
            model.train()
            optimizer.zero_grad()
            user, item, labels_list = data
            # trust2
            unique_user = set(user)
            path_index = []
            for u in unique_user:
                path_index.extend(model.user_path_indx[u.item()])
            if len(path_index) > trust_batch_size:
                path_index = random.sample(path_index, trust_batch_size)

            # multi_task
            loss1, loss2 = model(user=trans_to_cuda(user), item=trans_to_cuda(item),
                                 labels_list=trans_to_cuda(labels_list), trust_data=model.trust_train_data,
                                 slice_indices=slice_indices, flag=0)
            T = len(path_index)
            n_rec = 5
            T_rec = len(user)
            precision1 = torch.exp(-2 * model.task_weights[0])
            precision2 = torch.exp(-2 * model.task_weights[1])
            # multi_task
            loss = precision1 * loss1 + precision2 * loss2 + 2 * (n_rec + 1) * T_rec * model.task_weights[0] + T * model.task_weights[1]
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += float(loss)
        t2 = time()
        train_time = t2 - t1
        log.record('%d,%.5f,%.5f,%.5f,%.5f' % (epoch, precision1, precision2, total_loss1, total_loss2)) # 画图文件
        model.eval()
        # rec test
        users_to_test = list(data_generator.test_set.keys())
        ret = rec_test(model, users_to_test, drop_flag=True)  # 测试 lly
        perf_str = 'Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
            epoch, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1],
            ret['ndcg'][2])
        log.record(perf_str)
        if ret['recall'][0] > best_recall[0]:
            best_recall, best_iter[0] = ret['recall'], epoch
        if ret['ndcg'][0] > best_ndcg[0]:
            best_ndcg, best_iter[1] = ret['ndcg'], epoch
        # trust2 test
        hit10, hit20, hit50, ndcg10, ndcg20, ndcg50 = trust_test5(model, model.trust_test_data)
        perf_str = 'Trust:Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
            epoch, hit10, hit20, hit50, ndcg10, ndcg20, ndcg50)
        log.record(perf_str)
        if hit10 >= best_result[0]:
            best_result[:3] = [hit10, hit20, hit50]
            best_epoch[0] = epoch
        if ndcg10 >= best_result[3]:
            best_result[3:] = [ndcg10, ndcg20, ndcg50]
            best_epoch[1] = epoch

    log.record("--- Train Best ---")
    best_rec = 'Rec:  recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
        best_recall[0], best_recall[1], best_recall[2], best_ndcg[0], best_ndcg[1], best_ndcg[2])
    best_trust = 'Trust:recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
        best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5])
    log.record(best_rec)
    log.record(best_trust)


if __name__ == '__main__':

    data_generator.print_statistics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    if args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        # print('use the normalized adjacency matrix')
    else:
        config['norm_adj'] = mean_adj + sparse.eye(mean_adj.shape[0])
        # print('use the mean adjacency matrix')

    # trust
    train_data = pickle.load(open(args.data_path + args.dataset + '/trust/train.txt', 'rb'))
    test_data = pickle.load(open(args.data_path + args.dataset + '/trust/test2.txt', 'rb'))
    user_path_indx = defaultdict(list)
    path = train_data[0]
    for i, p in zip(range(len(path)), path):
        u = p[0]
        user_path_indx[u].append(i)
    all_path_index = set(range(len(path)))
    trust_train_data = Data(train_data, config['n_users'], shuffle=False)
    trust_test_data = Data(test_data, config['n_users'], shuffle=False, test=True)
    config['trust_train_data'] = trust_train_data
    config['trust_test_data'] = trust_test_data
    config['user_path_indx'] = user_path_indx
    config['all_path_index'] = all_path_index

    Engine = Model_Wrapper(data_config=config, device=device).to(device)

    # multi
    optimizer = torch.optim.Adam(Engine.parameters(), lr=args.lr)
    train(Engine, optimizer)
