import os
from gt_parser import parse_args
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

from utility1.Logging import Logging
from utility1.UV_Encoders_LN import UV_Encoder
from utility1.UV_Aggregators_LN import UV_Aggregator
from utility1.Social_Encoders_LN import Social_Encoder
from utility1.Social_Aggregators_LN import Social_Aggregator
from utility1.batch_test import test_one_user, test_one_user2, Ks
from utility2.layers import GraphAttentionLayer
from utility2.utils import Data

import torch
from torch import nn
import torch.nn.functional as F
import math
import random
import pickle
import numpy as np
from time import time
import multiprocessing
import torch.utils.data
cores = multiprocessing.cpu_count()
from functools import partial
from collections import defaultdict

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
log_path = os.path.join(os.getcwd(), 'log/%s_ats.log' % (args.dataset))
log = Logging(log_path)

class Graph_Session(nn.Module):

    def __init__(self, enc_u, enc_v_history, u2e, args):
        super(Graph_Session, self).__init__()
        # trust2
        self.hidden_size = args.embed_dim
        self.batch_size = args.batch_size
        self.nonhybrid = args.nonhybrid
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()

        self.reset_parameters()

        # multi head
        self.in_att = [GraphAttentionLayer(self.hidden_size, concat=True) for _ in range(args.nb_heads)]
        for i, attention in enumerate(self.in_att):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(self.hidden_size, concat=False)
        self.w = nn.Parameter(torch.zeros(size=(args.nb_heads * self.hidden_size, self.hidden_size)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)

        # rec
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim
        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.BCEWithLogitsLoss()

        # multi_task
        self.task_weights = nn.Parameter(torch.FloatTensor([0.0, 0.0]))

        # expert_s------------------------------------
        self.bn30 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn40 = nn.BatchNorm1d(16, momentum=0.5)
        self.w_uv10 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv20 = nn.Linear(self.embed_dim, 16)

        self.gate_w = nn.Parameter(torch.zeros(size=(32, 2)))
        nn.init.xavier_uniform_(self.gate_w.data, gain=1)
        
        self.att_t = nn.Parameter(torch.zeros(size=(2 * self.embed_dim, 2)))
        nn.init.xavier_normal_(self.att_t.data, gain=1)
        # -----------------------------------------

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, inputs, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            p_a = self.linear_transform(torch.cat([a, ht], 1))
        # ------------------------------
        p_i_all = u2e.weight[inputs]
        p_i = p_i_all * mask.unsqueeze(2)
        p_maxpool = torch.max(p_i, dim=1)[0]
        
        att = torch.softmax(torch.matmul(torch.cat([p_a, p_maxpool], 1), self.att_t), 1)  # [B,2H] x [2H,2] = [B,2]
        a = p_a * att[:, 0].unsqueeze(1) + p_maxpool * att[:, 1].unsqueeze(1)
        # ------------------------------
        b = u2e.weight[:-1]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, nodes_u, nodes_v, labels_list, i, data, flag=0):
        ''' two att no tower '''
        # rec
        if (flag == 1) or (flag == 0):
            embeds_u = self.enc_u(nodes_u)
            embeds_v = self.enc_v_history(nodes_v)
            x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
            x_u = F.dropout(x_u, training=self.training)
            x_u = self.w_ur2(x_u)
            x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
            x_v = F.dropout(x_v, training=self.training)
            x_v = self.w_vr2(x_v)
            x_uv = torch.cat((x_u, x_v), 1)
            x = F.relu(self.bn3(self.w_uv1(x_uv)))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.bn4(self.w_uv2(x)))
            x = F.dropout(x, training=self.training)

            # ----------------------------------------------
            x_uv0 = torch.cat((embeds_u, embeds_v), 1)  # [256, 128]
            x0 = F.relu(self.bn30(self.w_uv10(x_uv0)))
            x0 = F.dropout(x0, training=self.training)  # [256, 64]
            x0 = F.relu(self.bn40(self.w_uv20(x0)))
            x0 = F.dropout(x0, training=self.training)  # [256, 16]

            gate = F.softmax(torch.mm(torch.cat([x, x0], 1), self.gate_w), dim=1)
            x_f = gate[:, 0].unsqueeze(0).t() * x + gate[:, 1].unsqueeze(0).t() * x0
            # -----------------------------------------------------
            scores_r = self.w_uv3(x_f).squeeze()
            if flag == 1: return scores_r
            loss1 = self.criterion(scores_r, labels_list)
        # trust
        if (flag == 2) or (flag == 0):
            if flag == 0:
                inputs, mask, targets = data.get_slice(i)
            else:
                inputs, mask, targets, negs = data.get_slice(i)
                negs = torch.Tensor(negs).long().cuda()
            inputs = torch.Tensor(inputs).long().cuda()
            mask = torch.Tensor(mask).long().cuda()
            seq_l = torch.sum(mask, 1)
            # multi head
            mul_seq = torch.cat([att(u2e.weight, inputs, seq_l) for att in self.in_att], dim=2).cuda()
            mul_seq_c = torch.cat([mul_seq[i] for i in range(mul_seq.size()[0])], dim=0)
            mul_one = torch.mm(mul_seq_c, self.w)
            mul_one = F.elu(mul_one)
            seq_hidden_att = self.out_att(u2e.weight, mul_one.view(mul_seq.size()[0], mul_seq.size()[1], self.hidden_size), seq_l).cuda()
            scores = self.compute_scores(seq_hidden_att, inputs, mask)
            if (flag == 2):
                return scores, negs
            loss2 = self.loss_function(scores, torch.Tensor(targets).long().cuda())
        return loss1, loss2


def trust_test(model, test_data):
    l = test_data.length
    recall10, recall20, recall50, ndcg10, ndcg20, ndcg50 = 0, 0, 0, 0, 0, 0
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        trust_scores, trust_targets = model(None, None, None, i, test_data, 2)
        for score, target in zip(trust_scores, trust_targets):
            si = score[target].topk(50)[1].cpu().detach().numpy()
            re = test_one_user2([len(target)-1], si)
            recall10 += re['recall'][0]
            recall20 += re['recall'][1]
            recall50 += re['recall'][2]
            ndcg10 += re['ndcg'][0]
            ndcg20 += re['ndcg'][1]
            ndcg50 += re['ndcg'][2]

    return recall10 / l, recall20 / l, recall50 / l, ndcg10 / l, ndcg20 / l, ndcg50 / l


def train(model, device, train_loader, train_data2, user_path_indx, alluser_path_index, trust_batch_size, epoch,
          optimizer):
    model.train()
    running_loss = 0.0
    avail_path = alluser_path_index
    total_loss1, total_loss2 = 0., 0.
    for data_r in train_loader:
        optimizer.zero_grad()
        batch_nodes_u, batch_nodes_v, labels_list = data_r
        unique_user = set(batch_nodes_u.numpy().tolist())
        path_index = []
        for u in unique_user:
            path_index.extend(user_path_indx[u])

        if len(path_index) > trust_batch_size * 3:
            path_index = random.sample(path_index, trust_batch_size * 3)


        loss1, loss2 = model(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device),
                             np.array(list(path_index), dtype=int), train_data2, 0)


        # multi
        T = len(path_index)
        n_rec = 5
        T_rec = len(batch_nodes_u)
        precision1 = torch.exp(-2 * model.task_weights[0])
        precision2 = torch.exp(-2 * model.task_weights[1])
        loss = precision1 * loss1 + precision2 * loss2 + 2 * (n_rec + 1) * T_rec * model.task_weights[0] + T * \
               model.task_weights[1]
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
    train_rec = '%d,%.5f,%.5f,%.5f,%.5f' % (epoch, precision1, precision2, total_loss1, total_loss2)
    log.record(train_rec)
    return train_rec


def Test(model, epoch, device, test_loader, test_data2, best_rec, best_epoch, best_result):
    result_rec = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
                  'ndcg': np.zeros(len(Ks)), 'hit_ratio': np.zeros(len(Ks)),
                  'auc': 0.}
    model.eval()
    with torch.no_grad():
        # rec  
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v, None, None, None, 1)  # 预测
            all_items = list(test_v.data.cpu().numpy())
            predict_rec = list(val_output.data.cpu().numpy())
            re = test_one_user([all_items[99]], all_items, predict_rec)
            l = len(test_loader)
            result_rec['recall'] += re['recall'] / l
            result_rec['ndcg'] += re['ndcg'] / l
        if result_rec['recall'][0] > best_rec[0]:
            best_rec[:3] = result_rec['recall']
            best_epoch[0] = epoch
        if result_rec['ndcg'][0] > best_rec[3]:
            best_rec[3:] = result_rec['ndcg']
            best_epoch[1] = epoch
            # trust
        recall10, recall20, recall50, ndcg10, ndcg20, ndcg50 = trust_test(model, test_data2)
        if recall10 >= best_result[0]:
            best_result[:3] = [recall10, recall20, recall50]
            best_epoch[2] = epoch
        if ndcg10 >= best_result[3]:
            best_result[3:] = [ndcg10, ndcg20, ndcg50]
            best_epoch[3] = epoch

    perf_rec = 'Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
        epoch, result_rec['recall'][0], result_rec['recall'][1], result_rec['recall'][2], result_rec['ndcg'][0],
        result_rec['ndcg'][1],
        result_rec['ndcg'][2])
    log.record(perf_rec)
    perf_str = 'Trust:Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
        epoch, recall10, recall20, recall50, ndcg10, ndcg20, ndcg50)
    log.record(perf_str)
    return perf_rec, perf_str, best_rec, best_epoch, best_result



def create_data():
    BATCH_SIZE = 256
    pool = multiprocessing.Pool(cores)
    all_user = list(history_u_lists.keys())
    batch_times = len(all_user) // BATCH_SIZE
    train_u, train_v, train_r = [], [], []
    for batch_id in range(batch_times):
        start = batch_id * BATCH_SIZE
        end = (batch_id + 1) * BATCH_SIZE
        batch_nodes_u = all_user[start:end]
        partial_work = partial(train_sample, hu=history_u_lists, ai=all_items)
        uvr_zip = pool.map(partial_work, batch_nodes_u)
        for one_zip in uvr_zip:
            train_u.extend(one_zip[0])
            train_v.extend(one_zip[1])
            train_r.extend(one_zip[2])
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u),
                                              torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    pool.close()
    return trainset



def train_sample(u, hu, ai):
    pos_items = hu[u]
    pos_num = len(pos_items)
    neg_num = 5 * pos_num
    all_num = pos_num + neg_num
    us = [u] * all_num
    opt_items = ai - set(pos_items)
    vs = random.sample(opt_items, neg_num)
    vs.extend(pos_items)
    rs = [0] * neg_num + [1] * pos_num
    return [us, vs, rs]


def process_data(num_users, num_items, history_u_lists, history_v_lists, social_adj_lists, device):
    u_v_l = []  # len+1
    u_v = []  # len
    u_u_l = []
    u_u = []
    for i in range(num_users):
        u_v_l.append([len(history_u_lists[i])])
        u_u_l.append([len(social_adj_lists[i])])
    u_v_l.append([0])
    u_u_l.append([0])
    u_v_l_max = max(u_v_l)[0]
    u_u_l_max = max(u_u_l)[0]
    for i in range(num_users):
        u_v.append(history_u_lists[i] + [num_items] * (u_v_l_max - len(history_u_lists[i])))
        u_u.append(list(social_adj_lists[i]) + [num_users] * (u_u_l_max - len(social_adj_lists[i])))
    u_v_l = torch.FloatTensor(u_v_l).to(device)
    u_u_l = torch.FloatTensor(u_u_l).to(device)
    u_v = torch.tensor(u_v).to(device)
    u_u = torch.tensor(u_u).to(device)

    v_u_l = []
    v_u = []
    for i in range(num_items):
        v_u_l.append([len(history_v_lists[i])])
    v_u_l.append([0])
    v_u_l_max = max(v_u_l)[0]
    for i in range(num_items):
        v_u.append(history_v_lists[i] + [num_users] * (v_u_l_max - len(history_v_lists[i])))
    v_u_l = torch.FloatTensor(v_u_l).to(device)
    v_u = torch.tensor(v_u).to(device)

    return u_v_l, v_u_l, u_u_l, u_v, v_u, u_u


if __name__ == "__main__":
    # Training settings

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = args.embed_dim
    dir_data = '../data/{}/'.format(args.dataset) + 'rec/{}_t_dataset'.format(args.dataset)

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, \
    train_u, train_v, train_r, test_u, test_v, test_r, train_u1, train_u2, \
    train_tr, test_u1, test_u2, test_tr, social_adj_lists, ratings_list = \
        pickle.load(data_file)

    # rec
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    all_items = set(history_v_lists.keys())
    all_users = set(history_u_lists.keys())
    num_users = max(all_users) + 1
    num_items = max(all_items) + 1
    intro_prt = args.dataset + "\nuse: " + str(num_users) + "\nitem: " + str(num_items) + "\n----------------"
    log.record(intro_prt)
    # trust
    train_data2 = pickle.load(open('../data/' + args.dataset + '/trust/train.txt', 'rb'))
    test_data2 = pickle.load(open('../data/' + args.dataset + '/trust/test2.txt', 'rb'))
    user_path_indx = defaultdict(list)
    path = train_data2[0]
    alluser_path_index = set(range(len(path)))
    for i, p in zip(range(len(path)), path):
        u = p[0]
        user_path_indx[u].append(i)
    train_data2 = Data(train_data2, num_users, shuffle=False)
    test_data2 = Data(test_data2, num_users, shuffle=False, test=True)

    u2e = nn.Embedding(num_users + 1, embed_dim).to(device)
    v2e = nn.Embedding(num_items + 1, embed_dim).to(device)

    u_v_l, v_u_l, u_u_l, u_v, v_u, u_u = process_data(num_users, num_items, history_u_lists, history_v_lists,
                                                      social_adj_lists, device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, u2e, u_v, v_u, u_v_l, v_u_l, uv=True)
    enc_u_history = UV_Encoder(u2e, agg_u_history, embed_dim)
    # neighobrs
    agg_u_social = Social_Aggregator(u2e, u_u, u_u_l)
    enc_u = Social_Encoder(enc_u_history, agg_u_social, embed_dim)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, u2e, u_v, v_u, u_v_l, v_u_l, uv=False)
    enc_v_history = UV_Encoder(v2e, agg_v_history, embed_dim)

    # model
    model = Graph_Session(enc_u, enc_v_history, u2e, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)



    best_rec, best_epoch = [0, 0, 0, 0, 0, 0], [-1, -1, -1, -1]
    best_result = [0, 0, 0, 0, 0, 0]
    for epoch in range(1, args.epochs + 1):
        t1 = time()
        trainset = create_data()
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
        trust_batch_size = len(alluser_path_index) // len(train_loader)
        train_prt = train(model, device, train_loader, train_data2, user_path_indx, alluser_path_index,
                              trust_batch_size, epoch, optimizer)
        t2 = time()
        test_rec, test_str, best_rec, best_epoch, best_result = Test(model, epoch, device, test_loader, test_data2,
                                                                         best_rec, best_epoch, best_result)
        t3 = time()

    log.record("--- Train Best ---")
    best_rec = 'Rec:  recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
            best_rec[0], best_rec[1], best_rec[2], best_rec[3], best_rec[4], best_rec[5])
    best_trust = 'Trust:recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
            best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5])
    log.record(best_rec)
    log.record(best_trust)