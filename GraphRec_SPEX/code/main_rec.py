import torch
import torch.nn as nn
import pickle
from time import time
from utility1.UV_Encoders_LN import UV_Encoder
from utility1.UV_Aggregators_LN import UV_Aggregator
from utility1.Social_Encoders_LN import Social_Encoder
from utility1.Social_Aggregators_LN import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from gt_parser import parse_args
import os
from utility1.batch_test import *
import random
import multiprocessing

cores = multiprocessing.cpu_count()
from functools import partial


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history):
        super(GraphRec, self).__init__()
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

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)
        # rec
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)
        # rec
        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores_r = self.w_uv3(x)
        return scores_r.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores_r = self.forward(nodes_u, nodes_v)
        return self.criterion(scores_r, labels_list)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
    train_rec = '%d,%.5f' % (epoch, running_loss)
    print(train_rec)
    return train_rec


def Test(model, epoch, device, test_loader, best_rec, best_epoch):
    result_rec = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
                  'ndcg': np.zeros(len(Ks)), 'hit_ratio': np.zeros(len(Ks)),
                  'auc': 0.}

    model.eval()
    with torch.no_grad():
        # rec
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)  # 预测
            test_all_items = list(test_v.data.cpu().numpy())
            predict_rec = list(val_output.data.cpu().numpy())
            re = test_one_user([test_all_items[99]], test_all_items, predict_rec)
            l = len(test_loader)
            result_rec['precision'] += re['precision'] / l
            result_rec['recall'] += re['recall'] / l
            result_rec['ndcg'] += re['ndcg'] / l
            result_rec['hit_ratio'] += re['hit_ratio'] / l
            result_rec['auc'] += re['auc'] / l
        if result_rec['recall'][0] > best_rec[0]:
            best_rec[:3] = result_rec['recall']
            best_epoch[0] = epoch
        if result_rec['ndcg'][0] > best_rec[3]:
            best_rec[3:] = result_rec['ndcg']
            best_epoch[1] = epoch
    perf_rec = 'Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
        epoch, result_rec['recall'][0], result_rec['recall'][1], result_rec['recall'][2], result_rec['ndcg'][0],
        result_rec['ndcg'][1],
        result_rec['ndcg'][2])
    print(perf_rec)
    return perf_rec, best_rec, best_epoch


# 生成训练数据
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
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    pool.close()
    return train_loader



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
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    dir_data = '../data/{}/'.format(args.dataset) + 'rec/{}_t_dataset'.format(args.dataset)

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, train_u1, train_u2, train_tr, test_u1, test_u2, test_tr, social_adj_lists, ratings_list = pickle.load(
        data_file)

    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    num_users = max(history_u_lists.keys()) + 1
    num_items = max(history_v_lists.keys()) + 1
    num_ratings = max(ratings_list.keys()) + 1
    all_items = set(history_v_lists.keys())
    intro_prt = args.dataset + "\nuse: " + str(num_users) + "\nitem: " + str(num_items) + "\n----------------"
    print(intro_prt)

    u2e = nn.Embedding(num_users + 1, embed_dim).to(device)  # lly
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
    graphrec = GraphRec(enc_u, enc_v_history).to(device)
    optimizer = torch.optim.Adam(graphrec.parameters(), lr=args.lr)
    best_rec, best_epoch = [0, 0, 0, 0, 0, 0], [-1, -1]
    for epoch in range(1, args.epochs + 1):
        t1 = time()
        train_loader = create_data()
        train_prt = train(graphrec, device, train_loader, optimizer, epoch)
        t2 = time()
        test_prt, best_rec, best_epoch = Test(graphrec, epoch, device, test_loader, best_rec, best_epoch)
        t3 = time()
        # print("Train:[%.1fs],Test:[%.1fs]" % (t2 - t1, t3 - t2))
    print("--- Train Best ---")
    best_rec = 'recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
        best_rec[0], best_rec[1], best_rec[2], best_rec[3], best_rec[4], best_rec[5])
    print(best_rec)
