import os
from parser import parse_args
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
import pickle
import random
import math
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import torch.utils.data
from collections import defaultdict
from utility1.Logging import Logging
from utility1.load_data import Data as D1, get_train_instances
from utility1.batch_test import test_rec, test_trust5
from utility2.layers import GraphAttentionLayer
from utility2.utils import Data as D2


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



class SAMN_TRUST(nn.Module):

    def __init__(self, user_num, item_num, args):
        super(SAMN_TRUST, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = args.embedding_size

        # trust
        self.batch_size = args.batch_size
        self.nonhybrid = args.nonhybrid
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_size * 2, self.embedding_size, bias=True)
        self.reset_parameters()

        self.in_att = [GraphAttentionLayer(self.embedding_size, concat=True) for _ in range(args.nb_heads)]
        for i, attention in enumerate(self.in_att):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(self.embedding_size, concat=False)
        self.w = nn.Parameter(torch.zeros(size=(args.nb_heads * self.embedding_size, self.embedding_size)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.criterion2 = nn.CrossEntropyLoss()  # 损失函数

        # rec
        self.attention_size = args.attention_size
        self.memory_size = args.memory_size
        self.i_bias = nn.Parameter(torch.FloatTensor([0.0] * self.item_num))
        self.Key = nn.Parameter(torch.FloatTensor(self.embedding_size, self.memory_size).uniform_(-0.1, 0.1))
        self.Mem = nn.Parameter(torch.FloatTensor(self.memory_size, self.embedding_size).uniform_(1.0, 1.0))
        self.WA = nn.Parameter(torch.FloatTensor(self.embedding_size, self.attention_size).uniform_(-0.1, 0.1))
        self.BA = nn.Parameter(torch.zeros(self.attention_size))
        self.U_omega = nn.Parameter(torch.FloatTensor(self.attention_size, 1).uniform_(-0.1, 0.1))
        self.dropout = nn.Dropout(p=args.dp)
        self.criterion1 = nn.BCEWithLogitsLoss()

        # share
        self.u2e_r = nn.Parameter(torch.FloatTensor(user_num + 1, self.embedding_size).uniform_(-0.1, 0.1))
        self.iidW = nn.Parameter(torch.FloatTensor(item_num, self.embedding_size).uniform_(-0.1, 0.1))
        # trust
        self.u2e_t = nn.Parameter(torch.FloatTensor(user_num + 1, self.embedding_size).uniform_(-0.1, 0.1))
        # share
        self.mlp = nn.Sequential()
        self.mlp.add_module("Linear_layer_s", nn.Linear(self.embedding_size, self.embedding_size))
        self.mlp.add_module("Relu_layer_s", nn.ReLU(inplace=True))

        self.mlp_r = nn.Sequential()
        self.mlp_r.add_module("Linear_layer_r", nn.Linear(self.embedding_size * 2, self.embedding_size))
        self.mlp_r.add_module("Relu_layer_r", nn.ReLU(inplace=True))

        self.mlp_t = nn.Sequential()
        self.mlp_t.add_module("Linear_layer_t", nn.Linear(self.embedding_size * 2, self.embedding_size))
        self.mlp_t.add_module("Relu_layer_t", nn.ReLU(inplace=True))

        # multi_task
        self.task_weights = nn.Parameter(torch.FloatTensor([0.5, 0.5]))


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_u, input_i, label, input_uf, i, data, flag):
        share_u_r = self.mlp(self.u2e_r)
        share_u_t = self.mlp(self.u2e_t)
        self.uidW_r = self.mlp_r(torch.cat([self.u2e_r, share_u_r], dim=1))
        self.uidW_t = self.mlp_t(torch.cat([self.u2e_t, share_u_t], dim=1))

        if (flag == 1) or (flag == 0):
            self.uid = self.uidW_r[input_u]
            self.iid = self.iidW[input_i]
            self.uid = self.uid.reshape([-1, self.embedding_size])
            self.iid = self.iid.reshape([-1, self.embedding_size])
            self.i_b = self.i_bias[input_i]
            # memory_attention
            self.frien_embedding = self.uidW_r[input_uf]
            self.frien_num = (input_uf.eq(self.user_num) == False).float()
            self.frien_embedding = torch.einsum('ab,abc->abc', [self.frien_num, self.frien_embedding])
            self.uid_n = torch.nn.functional.normalize(self.uid, p=2, dim=1)
            self.frien_embedding_n = torch.nn.functional.normalize(self.frien_embedding, p=2, dim=2)
            self.cross_friend = torch.einsum('ac,abc->abc', self.uid_n, self.frien_embedding_n)
            self.att_key = torch.einsum('abc,ck->abk', [self.cross_friend, self.Key])
            self.att_mem = F.softmax(self.att_key, dim=0)
            self.att_mem = torch.einsum('ab,abc->abc', [self.frien_num, self.att_mem])
            self.frien_f1 = torch.einsum('abc,ck->abk', [self.att_mem, self.Mem])
            self.frien_f2 = torch.mul(self.frien_f1, self.frien_embedding)
            # friend_attention
            self.frien_j = torch.exp(torch.einsum('abc,ck->abk', F.relu(torch.einsum('abc,ck->abk', self.frien_f2, self.WA) + self.BA),self.U_omega))
            self.frien_j = torch.einsum('ab,abc->abc', [self.frien_num, self.frien_j])
            self.frien_sum = torch.sum(self.frien_j, dim=1).unsqueeze(1) + 1e-8
            self.frien_w = torch.div(self.frien_j, self.frien_sum)
            self.friend = torch.sum(torch.mul(self.frien_w, self.frien_f2), dim=1)
            self.friend = self.dropout(self.friend)
            self.user = self.uid + self.friend
            self.score = torch.sum(torch.mul(self.user, self.iid), 1) + self.i_b
            if flag == 1: return self.score
            loss1 = self.criterion1(self.score, label)
        if (flag == 2) or (flag == 0):
            if flag == 0:
                inputs, mask, targets = data.get_slice(i)
            else:
                inputs, mask, targets, negs = data.get_slice(i)
                negs = torch.Tensor(negs).long().cuda()

            inputs = torch.Tensor(inputs).long().cuda()
            mask = torch.Tensor(mask).long().cuda()
            targets = torch.Tensor(targets).long().cuda()
            seq_l = torch.sum(mask, 1)

            # multi head
            mul_seq = torch.cat([att(self.uidW_t, inputs, seq_l) for att in self.in_att], dim=2).cuda()
            mul_seq_c = torch.cat([mul_seq[i] for i in range(mul_seq.size()[0])], dim=0)
            mul_one = torch.mm(mul_seq_c, self.w)
            mul_one = F.elu(mul_one)
            seq_hidden = self.out_att(self.uidW_t,
                                          mul_one.view(mul_seq.size()[0], mul_seq.size()[1], self.embedding_size),
                                          seq_l).cuda()
            scores = self.compute_scores(seq_hidden, mask)
            if (flag == 2): return scores, negs
            loss2 = self.criterion2(scores, targets)

        return loss1, loss2

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.uidW_t[:-1]
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores


def train(model, optimizer, train_loader, epoch):
    trust_batch_size = len(alluser_path_index) // len(train_loader)
    p1, p2, t_loss1, t_loss2 = 0.0, 0.0, 0.0, 0.0
    for data in train_loader:
        optimizer.zero_grad()
        batch_user, batch_item, label, batch_uf = data
        unique_user = set(batch_user.numpy().tolist())
        path_index = []
        for u in unique_user:
            path_index.extend(user_path_indx[u])
        if len(path_index) > trust_batch_size:
            path_index = random.sample(path_index, trust_batch_size)
        loss1, loss2 = model(batch_user.cuda(), batch_item.cuda(), label.cuda(), batch_uf.cuda(),np.array(list(path_index), dtype=int), train_data2, 0)
        # multi
        T = len(path_index)
        n_rec = 5
        T_rec = len(batch_user)
        precision1 = torch.exp(-2 * model.task_weights[0])
        precision2 = torch.exp(-2 * model.task_weights[1])
        loss = precision1 * loss1 + precision2 * loss2 + 2 * (n_rec + 1) * T_rec * model.task_weights[0] + T * model.task_weights[1]
        loss.backward()
        optimizer.step()

        p1, p2 = precision1, precision2
        t_loss1 += loss1.item()
        t_loss2 += loss2.item()
    log.record('%d,%.5f,%.5f,%.5f,%.5f' % (epoch, p1, p2, t_loss1, t_loss2))  # 画图文件


if __name__ == '__main__':
    # rec
    data_r = D1(path=args.data_root + args.dataset + '/rec/')
    n_users, n_items, tp_test, tp_train = data_r.n_users, data_r.n_items, data_r.tp_test, data_r.tp_train
    test_item, neg_item, tfset, max_friend = data_r.test_item, data_r.neg_item, data_r.tfset, data_r.max_friend
    u_train = np.array(tp_train['uid'], dtype=np.int32)
    i_train = np.array(tp_train['sid'], dtype=np.int32)
    # trust
    train_data2 = pickle.load(open(args.data_root + args.dataset + '/trust/train.txt', 'rb'))
    test_data2 = pickle.load(open(args.data_root + args.dataset + '/trust/test2.txt', 'rb'))
    user_path_indx = defaultdict(list)
    path = train_data2[0]
    alluser_path_index = set(range(len(path)))
    for i, p in zip(range(len(path)), path):
        u = p[0]
        user_path_indx[u].append(i)
    train_data2 = D2(train_data2, n_users, shuffle=True)
    test_data2 = D2(test_data2, n_users, shuffle=False, test=True)

    model = SAMN_TRUST(n_users, n_items, args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rec_bres = [0, 0, 0, 0, 0, 0]
    trust_bres = [0, 0, 0, 0, 0, 0]
    for epoch in range(args.epochs):
        model.train()
        trainset = get_train_instances(u_train, i_train, tfset, n_items, n_users, max_friend)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        train(model, optimizer, train_loader, epoch)

        model.eval()
        result = test_rec(model, test_item, neg_item, n_users, max_friend, tfset)
        if result['recall'][0] >= rec_bres[0]:
            rec_bres[:3] = result['recall']
        if result['ndcg'][0] >= rec_bres[3]:
            rec_bres[3:] = result['ndcg']
        log.record('Epoch[%d], Rec: recall=[%.5f, %.5f, %.5f],  ndcg=[%.5f, %.5f, %.5f]' % (epoch, result['recall'][0], result['recall'][1], result['recall'][2], result['ndcg'][0], result['ndcg'][1],result['ndcg'][2]))

        result2 = test_trust5(model, test_data2)
        if result2['recall'][0] >= trust_bres[0]:
            trust_bres[:3] = result2['recall']
        if result2['ndcg'][0] >= trust_bres[3]:
            trust_bres[3:] = result2['ndcg']
        log.record('Epoch[%d], Tru: recall=[%.5f, %.5f, %.5f],  ndcg=[%.5f, %.5f, %.5f]' % (epoch, result2['recall'][0], result2['recall'][1], result2['recall'][2], result2['ndcg'][0], result2['ndcg'][1],result2['ndcg'][2]))

    log.record("--- Train Best ---")
    best_rec = 'Rec : recall=[%.5f, %.5f, %.5f],  ndcg=[%.5f, %.5f, %.5f]' % (
    rec_bres[0], rec_bres[1], rec_bres[2], rec_bres[3], rec_bres[4],rec_bres[5])
    best_trust = 'Trust : recall=[%.5f, %.5f, %.5f],  ndcg=[%.5f, %.5f, %.5f]' % (
    trust_bres[0], trust_bres[1], trust_bres[2], trust_bres[3], trust_bres[4],trust_bres[5])
    log.record(best_rec)
    log.record(best_trust)

