import os
from trust_parser import trust_parse_args
args = trust_parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
import time
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from utility2.Logging import Logging
from utility2.utils import Data
from utility2.model import TRUST_PATH
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
log_path = os.path.join(os.getcwd(), 'log/%s_trust.log' % (args.dataset))
log = Logging(log_path)
############################## PREPARE DATASET ##########################

trust_train_data = pickle.load(open(args.data_root + args.dataset + '/trust/train.txt', 'rb'))
trust_test_data = pickle.load(open(args.data_root + args.dataset + '/trust/test2.txt', 'rb'))
user_path_indx = defaultdict(list)
if args.dataset == "epinion2":
    args.user_num = 12645
elif args.dataset == "weibo":
    args.user_num = 6812
elif args.dataset == "twitter":
    args.user_num = 8930
train_data2 = Data(trust_train_data, args.user_num, shuffle=True)
test_data2 = Data(trust_test_data, args.user_num, shuffle=False,test=True)

u_f2 = pickle.load(open(args.data_root + args.dataset + '/trust/fs.dic', 'rb'))

########################### CREATE MODEL #################################
model = TRUST_PATH(args)
model = model.cuda()

trust_loss_function = nn.CrossEntropyLoss()
trust_loss_function2 = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
########################### TRAINING #####################################

best_result = [0, 0, 0, 0, 0, 0]
best_epoch = [0, 0]

for epoch in range(args.epochs):
    t0 = time.perf_counter()

    total_loss = 0.0
    model.train()  # Enable dropout (if have).
    slice = train_data2.generate_batch(256)
    for slice_indices in slice:
        trust_scores, trust_targets = model(None, None, slice_indices, train_data2, flag=0)
        loss = trust_loss_function(trust_scores, trust_targets)
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    t1 = time.perf_counter()
    log.record('%d, %.5f, %.5f' % (epoch, total_loss, t1 - t0))

    model.eval()
    recall10, recall20, recall50, ndcg10, ndcg20, ndcg50 = trust_test5(model, test_data2)
    t2 = time.perf_counter()
    log.record('Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f] Time=[%.4f]' % (
        epoch, recall10, recall20, recall50, ndcg10, ndcg20, ndcg50, t2-t1))
    if recall10 >= best_result[0]:
        best_result[:3] = [recall10, recall20, recall50]
        best_epoch[0] = epoch
    if ndcg10 >= best_result[3]:
        best_result[3:] = [ndcg10, ndcg20, ndcg50]
        best_epoch[1] = epoch

best_trust = 'BEST: recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4],best_result[5])
log.record(best_trust)
