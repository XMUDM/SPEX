import os
from lg_parser import parse_args_r
args = parse_args_r()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
import time
import pickle
import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from utility1.dataloader import LightTrainData
import utility1.dataloader as dataloader
import utility1.utils as utils
import utility1.model_expert_s as model
from utility1.batch_test import rec_test
from utility2.utils import Data
from utility2.batch_test_gnn import trust_test5

# ==============================
utils.set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==============================
# rec
dataset = dataloader.Loader(args)
train_dataset = LightTrainData(dataset.rec_train_data, dataset.m_item, dataset.train_mat)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# trust
train_data2 = pickle.load(open(args.data_path + args.dataset + '/trust/train.txt', 'rb'))
test_data2 = pickle.load(open('../data/' + args.dataset + '/trust/test2.txt', 'rb'))
user_path_indx = defaultdict(list)
path = train_data2[0]
for i, p in zip(range(len(path)), path):
    u = p[0]
    user_path_indx[u].append(i)
train_data2 = Data(train_data2, dataset.n_users, shuffle=False)
test_data2 = Data(test_data2, dataset.n_users, shuffle=False, test=True)
trust_batch_size = len(path) // len(train_loader)

Recmodel = model.LightGCN(args, dataset).to(device)
optimizer = torch.optim.Adam(Recmodel.parameters(), lr=args.lr)

def Train(train_loader, recommend_model, epoch, optimizer):
    train_loader.dataset.ng_sample()
    Recmodel = recommend_model
    Recmodel.train()
    total_loss = 0.0
    total_loss1, total_loss2 = 0., 0.
    for data in train_loader:
        # rec
        optimizer.zero_grad()
        user, item, label = data
        unique_user = set(user.numpy().tolist())
        path_index = []
        for u in unique_user:
            path_index.extend(user_path_indx[u])
        if len(path_index) > trust_batch_size:
            path_index = random.sample(path_index, trust_batch_size)


        loss1, loss2 = Recmodel(users=user.to(device), items=item.to(device), labels=label.to(device), slice_indices=np.array(list(path_index), dtype=int), trust_data=train_data2, flag=0)
        # multi
        T = len(path_index)
        n_rec = 5
        T_rec = len(user)
        precision1 = torch.exp(-2 * Recmodel.task_weights[0])
        precision2 = torch.exp(-2 * Recmodel.task_weights[1])
        loss = loss1 + loss2
        loss.backward()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        optimizer.step()
    total_loss = total_loss2 + total_loss1
    print('%d,%.5f,%.5f,%.5f,%.5f' % (epoch, precision1, precision2, total_loss1, total_loss2))

def Test(dataset, Recmodel, epoch, best_recall, best_ndcg, best_iter, best_result):
    # rec
    dataset: utils.BasicDataset
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    testRatings = dataset.testRatings
    testNegatives = dataset.testNegatives
    with torch.no_grad():
        # rec
        ret = rec_test(Recmodel, testRatings, testNegatives)  # lly
        perf_str = 'Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
            epoch, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1],
            ret['ndcg'][2])
        print(perf_str)
        if ret['recall'][0] > best_recall[0]:
            best_recall, best_iter[0] = ret['recall'], epoch
        if ret['ndcg'][0] > best_ndcg[0]:
            best_ndcg, best_iter[1] = ret['ndcg'], epoch
        # trust
        recall10, recall20, recall50, ndcg10, ndcg20, ndcg50 = trust_test5(Recmodel, test_data2)
        perf_str = 'Trust:Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
            epoch, recall10, recall20, recall50, ndcg10, ndcg20, ndcg50)
        print(perf_str)
        if recall10 >= best_result[0]:
            best_result[:3] = [recall10, recall20, recall50]
        if ndcg10 >= best_result[3]:
            best_result[3:] = [ndcg10, ndcg20, ndcg50]
    return best_recall, best_ndcg, best_iter, best_result


best_recall, best_ndcg, best_iter = [0, 0, 0], [0, 0, 0], [0, 0]
best_result = [0, 0, 0, 0, 0, 0]

for epoch in range(args.epochs):
    start = time.time()
    Train(train_loader, Recmodel, epoch, optimizer)
    best_recall, best_ndcg, best_iter, best_result = Test(dataset, Recmodel, epoch, best_recall, best_ndcg, best_iter, best_result)
    end = time.time()
    train_time = end - start

print("--- Train Best ---")
best_rec = 'Rec:  recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (best_recall[0], best_recall[1], best_recall[2], best_ndcg[0], best_ndcg[1],best_ndcg[2])
best_trust = 'Trust:recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4],best_result[5])
print(best_rec)
print(best_trust)