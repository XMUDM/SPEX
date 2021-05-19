import os
from parser_a import parse_args
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
filename = 'log/%s_auto_%s_%s_0.005.log'
print(filename)
import pickle
import random
import numpy as np
from time import time
from collections import defaultdict
import torch
from utility.Logging import Logging
from utility.model_multi import DANSER
from utility.batch_test import rec_test1, to_gpuTensor
from utility.input import DataInput
from utility2.utils import Data as D2
from utility2.batch_test_gnn import test_trust5


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


setup_seed(2020)

# rec
workdir = args.data_root + args.dataset
with open(workdir + '/rec/dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    ui_p = pickle.load(f)
with open(workdir + '/rec/list.pkl', 'rb') as f:
    u_friend_list = pickle.load(f)
    u_read_list = pickle.load(f)
    uf_read_list = pickle.load(f)
    i_friend_list = pickle.load(f)
    i_read_list = pickle.load(f)
    if_read_list = pickle.load(f)
    i_link_list = pickle.load(f)
    user_count, item_count = pickle.load(f)

# trust
train_data2 = pickle.load(open(args.data_root + args.dataset + '/trust/train.txt', 'rb'))
test_data2 = pickle.load(open(args.data_root + args.dataset + '/trust/test2.txt', 'rb'))
user_path_indx = defaultdict(list)
path = train_data2[0]
alluser_path_index = set(range(len(path)))
for i, p in zip(range(len(path)), path):
    u = p[0]
    user_path_indx[u].append(i)
train_data2 = D2(train_data2, user_count, shuffle=True)
test_data2 = D2(test_data2, user_count, shuffle=False, test=True)

if __name__ == '__main__':
    model = DANSER(user_count, item_count, args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(os.getcwd(), filename % (args.dataset, args.embedding_size, args.keep_prob))
    log = Logging(log_path)
    log.record('Following will output the evaluation of the model:')

    best_recall, best_ndcg, best_r, best_t = 0., 0., 0., 0.
    trust_best_result = [0, 0, 0, 0, 0, 0]
    for epoch in range(args.epochs):
        # train
        loss_sum = 0.
        e = 0
        accumulation_steps = 4
        p1, p2, t_loss1, t_loss2 = 0.0, 0.0, 0.0, 0.0
        t1 = time()
        model.train()
        random.shuffle(train_set)
        train_dataset = DataInput(train_set, u_read_list, u_friend_list, uf_read_list, i_read_list, i_friend_list,
                                  if_read_list, i_link_list, args.train_batch_size, args.trunc_len, ui_p,
                                  set(range(item_count)), True)
        trust_train_batch_size = len(alluser_path_index) // train_dataset.epoch_size
        for u, i, l, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_linput, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput in train_dataset:
            unique_user = set(u.tolist())
            path_index = []
            for uu in unique_user:
                path_index.extend(user_path_indx[uu])
            if len(path_index) > trust_train_batch_size:
                path_index = random.sample(path_index, trust_train_batch_size)
            # for twitter
            if len(path_index) < trust_train_batch_size:
                path_index = random.sample(alluser_path_index, trust_train_batch_size)

            loss1, loss2 = model(to_gpuTensor(u), to_gpuTensor(i), to_gpuTensor(l), to_gpuTensor(u_readinput),
                                 to_gpuTensor(u_friendinput), to_gpuTensor(uf_readinput), to_gpuTensor(u_read_l),
                                 to_gpuTensor(u_friend_l), to_gpuTensor(uf_read_linput), to_gpuTensor(i_readinput),
                                 to_gpuTensor(i_friendinput), to_gpuTensor(if_readinput), to_gpuTensor(i_linkinput),
                                 to_gpuTensor(i_read_l), to_gpuTensor(i_friend_l), to_gpuTensor(if_read_linput),
                                 np.array(list(path_index), dtype=int), train_data2, 0)
            reg_loss = model.reg_loss()
            # multi
            T = len(path_index)
            n_rec = 5
            T_rec = len(u)
            precision1 = torch.exp(-2 * model.task_weights[0])
            precision2 = torch.exp(-2 * model.task_weights[1])
            loss = precision1 * loss1 + precision2 * loss2 + 2 * (n_rec + 1) * T_rec * model.task_weights[0] + T * model.task_weights[1] + 0.005*reg_loss
            optimizer.zero_grad()
            loss.backward()
            p1, p2 = precision1, precision2
            t_loss1 += loss1.item()
            t_loss2 += loss2.item()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 3, norm_type=2)  # max_norm 3,0.005,0.001
            optimizer.step()

        t2 = time()
        log.record('%d,%.5f,%.5f,%.5f,%.5f,%.5f' % (epoch, p1, p2, t_loss1, t_loss2, t2 - t1))  
        # test
        model.eval()
        test_dataset = DataInput(test_set, u_read_list, u_friend_list, uf_read_list, i_read_list, i_friend_list,
                                 if_read_list, i_link_list, args.test_batch_size, args.trunc_len, ui_p,
                                 set(range(item_count)), False)
        with torch.no_grad():
            ret = rec_test1(model, test_dataset)
            t3 = time()
            perf_rec = 'Rec: Recall=[%.5f,%.5f,%.5f], NDCG=[%.5f,%.5f,%.5f] Time:[%0.2f]' % (
            ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2],
            t3 - t2)
            log.record(perf_rec)
            if ret['recall'][0] > best_recall:
                best_recall, best_ndcg, best_r = ret['recall'][0], ret['ndcg'][0], epoch

            result2 = test_trust5(model, test_data2)
            if result2[0] >= trust_best_result[0]:
                trust_best_result[:3] = result2[:3]
                best_t = epoch
            if result2[3] >= trust_best_result[3]:
                trust_best_result[3:] = result2[3:]
            t4 = time()
            perf_trust = 'Trust: Recall=[%.5f,%.5f,%.5f], NDCG=[%.5f,%.5f,%.5f] Time:[%0.2f]' % (
            result2[0], result2[1], result2[2], result2[3], result2[4], result2[5], t4 - t3)
            log.record(perf_trust)

    log.record('Best Epoch[%d] Rec: Recall=[%.4f], NDCG=[%.4f]' % (best_r, best_recall, best_ndcg))
    log.record('Best Epoch[%d] Rec: Recall=[%.4f, %.4f, %.4f],  NDCG=[%.4f, %.4f, %.4f]' % (
    best_t, trust_best_result[0], trust_best_result[1], trust_best_result[2], trust_best_result[3],
    trust_best_result[4], trust_best_result[5]))
