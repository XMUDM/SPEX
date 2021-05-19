import os
from ncf_parser import ncf_parse_args
args = ncf_parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
import time
import random
import pickle
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utility.Logging import Logging
import utility.config as config
from utility.model_multi import NCF
from utility.data_utils import NCFData, load_all
from utility.batch_test import rec_test
from utility.gpuutil import trans_to_cuda
from utility2.utils import Data
from utility2.gnn_batch_test import trust_test5

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
log_path = os.path.join(os.getcwd(), 'log/%s_auto_%s.log' % (args.dataset, args.nb_heads))
log = Logging(log_path)
############################## PREPARE DATASET ##########################
# rec
train_data, testRatings, testNegatives, user_num, item_num, train_mat = load_all()
print("use:", user_num)
print("item:", item_num)
print("----------------")

train_dataset = NCFData(train_data, item_num, train_mat, args.num_ng, True)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
# trust
trust_train_data = pickle.load(open(args.data_root + args.dataset + '/trust/train.txt', 'rb'))
trust_test_data = pickle.load(open(args.data_root + args.dataset + '/trust/test2.txt', 'rb'))
user_path_indx = defaultdict(list)
path = trust_train_data[0]
for i, p in zip(range(len(path)), path):
    u = p[0]
    user_path_indx[u].append(i)
train_data2 = Data(trust_train_data, user_num, shuffle=False)
test_data2 = Data(trust_test_data, user_num, shuffle=False,test=True)
all_path_index = set(range(len(path)))

########################### CREATE MODEL #################################
# rec
model = NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, config.model, args)
model = trans_to_cuda(model)

rec_loss_function = nn.BCEWithLogitsLoss()
trust_loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

########################### TRAINING #####################################
# rec
best_recall, best_ndcg, best_iter = [0, 0, 0], [0, 0, 0], [0, 0]
# trust
best_result = [0, 0, 0, 0, 0, 0]
best_epoch = [0, 0]

for epoch in range(args.epochs):
    t0 = time.perf_counter()

    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0
    p1 = 0.0
    p2 = 0.0
    model.train()  # Enable dropout (if have).
    train_loader.dataset.ng_sample()
    avail_path_index = all_path_index
    trust_batch_size = len(avail_path_index) // len(train_loader)

    for data in train_loader:
        # rec
        user, item, label = data
        unique_user = set(user)
        path_index = []
        for u in unique_user:
            path_index.extend(user_path_indx[u.item()])

        if len(path_index) > trust_batch_size:
            path_index = random.sample(path_index, trust_batch_size)


        slice_indices = np.array(list(path_index), dtype=int)
        rec_prediction, trust_scores, trust_targets = model(user=trans_to_cuda(user), item=trans_to_cuda(item), slice_indices=slice_indices, trust_data=train_data2, flag=0)

        loss1 = rec_loss_function(rec_prediction, trans_to_cuda(label.float()))
        loss2 = trust_loss_function(trust_scores, trust_targets)

        # multi_task
        T = len(path_index)
        n_rec = 5
        T_rec = len(user)
        precision1 = torch.exp(-2 * model.log_vars[0])
        precision2 = torch.exp(-2 * model.log_vars[1])
        loss = precision1 * loss1 + precision2 * loss2 + 2 * (n_rec + 1) * T_rec * model.log_vars[0] + T * model.log_vars[1]
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        p1 = precision1
        p2 = precision2
        optimizer.step()
    log.record('%d,%.5f,%.5f,%.5f,%.5f' % (epoch,p1,p2,total_loss1,total_loss2))

    t1 = time.perf_counter()
    train_time = t1 - t0

    model.eval()
    # rec
    ret = rec_test(model, testRatings, testNegatives) 
    perf_str = 'Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
        epoch, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1],
        ret['ndcg'][2])
    log.record(perf_str)
    if ret['recall'][0] > best_recall[0]:
        best_recall, best_iter[0] = ret['recall'], epoch
    if ret['ndcg'][0] > best_ndcg[0]:
        best_ndcg, best_iter[1] = ret['ndcg'], epoch
    # trust
    hit10, hit20, hit50, ndcg10, ndcg20, ndcg50 = trust_test5(model, test_data2)
    perf_str = 'Trust:Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
        epoch, hit10, hit20, hit50, ndcg10, ndcg20, ndcg50)
    log.record(perf_str)
    if hit10 >= best_result[0]:
        best_result[:3] = [hit10, hit20, hit50]
        best_epoch[0] = epoch
    if ndcg10 >= best_result[3]:
        best_result[3:] = [ndcg10, ndcg20, ndcg50]
        best_epoch[1] = epoch
    t2 = time.perf_counter()
    test_time = t2 - t1
log.record("--- Train Best ---")
best_rec = 'Rec:  recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (best_recall[0], best_recall[1], best_recall[2], best_ndcg[0], best_ndcg[1],best_ndcg[2])
best_trust = 'Trust:recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4],best_result[5])
log.record(best_rec)
log.record(best_trust)

