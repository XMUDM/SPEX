import os
from ncf_parser import ncf_parse_args
args = ncf_parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utility.config as config
from utility.model import NCF
from utility.data_utils import NCFData, load_all
from utility.batch_test import test
from utility.gpuutil import trans_to_cuda

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(2020)

############################## PREPARE DATASET ##########################
# rec
train_data, testRatings, testNegatives, user_num, item_num, train_mat = load_all()
train_dataset = NCFData(train_data, item_num, train_mat, args.num_ng, True)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
print(args.dataset)
print("use:", user_num)
print("item:", item_num)
print("----------------")


########################### CREATE MODEL #################################
# rec
model = NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, config.model)
model = trans_to_cuda(model)

rec_loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

########################### TRAINING #####################################
# rec
best_recall, best_ndcg, best_iter = [0, 0, 0], [0, 0, 0], [0, 0]

for epoch in range(args.epochs):
    t0 = time.perf_counter()

    total_loss = 0.0
    model.train()  # Enable dropout (if have).
    train_loader.dataset.ng_sample()

    not_enogth = 0
    for data in train_loader:
        # rec
        user, item, label = data
        rec_prediction = model(user=trans_to_cuda(user), item=trans_to_cuda(item))
        loss = rec_loss_function(rec_prediction, trans_to_cuda(label.float()))
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    t1 = time.perf_counter()
    print('%d, %.5f, %.5f' % (epoch, total_loss, t1 - t0))

    model.eval()
    ret = test(model, testRatings, testNegatives)
    t2 = time.perf_counter()
    perf_str = 'Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f], Time:[%.4f]' % (
        epoch, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1],
        ret['ndcg'][2],t2 - t1)
    print(perf_str)
    if ret['recall'][0] > best_recall[0]:
        best_recall, best_iter[0] = ret['recall'], epoch
    if ret['ndcg'][0] > best_ndcg[0]:
        best_ndcg, best_iter[1] = ret['ndcg'], epoch
print("--- Train Best ---")
best_rec = 'recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (best_recall[0], best_recall[1], best_recall[2], best_ndcg[0], best_ndcg[1],best_ndcg[2])
print(best_rec)