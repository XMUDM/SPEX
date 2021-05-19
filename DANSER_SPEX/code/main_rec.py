import os
from parser_a import parse_args
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
import pickle
import random
import numpy as np
from time import time
import torch
from utility.Logging import Logging
from utility.model import DANSER
from utility.batch_test import rec_test3, to_gpuTensor
from utility.input import DataInput


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(2020)

# load data
workdir = args.data_root+args.dataset
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


if __name__ == '__main__':
    model = DANSER(user_count, item_count, args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(os.getcwd(), 'log/%s_rec_%s_%s.log' % (args.dataset, args.embedding_size,args.keep_prob))
    log = Logging(log_path)
    log.record('Following will output the evaluation of the model:')

    best_recall, best_ndcg, best_iter = 0., 0., 0.
    for epoch in range(args.epochs):
        # train
        model.train(mode=True)  
        random.shuffle(train_set)
        loss_sum = 0.
        e = 0
        accumulation_steps = 4
        t1 = time()
        dataset = DataInput(train_set, u_read_list, u_friend_list, uf_read_list, i_read_list, i_friend_list, if_read_list,i_link_list, args.train_batch_size, args.trunc_len, ui_p, set(range(item_count)),True)
        for u, i, l, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_linput, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput in dataset:
            optimizer.zero_grad()
            loss = model(to_gpuTensor(u),to_gpuTensor(i),to_gpuTensor(l), to_gpuTensor(u_readinput), to_gpuTensor(u_friendinput), to_gpuTensor(uf_readinput), to_gpuTensor(u_read_l), to_gpuTensor(u_friend_l), to_gpuTensor(uf_read_linput), to_gpuTensor(i_readinput), to_gpuTensor(i_friendinput), to_gpuTensor(if_readinput),  to_gpuTensor(i_linkinput), to_gpuTensor(i_read_l), to_gpuTensor(i_friend_l), to_gpuTensor(if_read_linput),1)
            loss_sum += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
        t2 = time()
        log.record("Epoch[%d] RecLoss:[%.5f] Time:[%.2f]"%(epoch,loss_sum,t2-t1))

        # test
        model.eval()
        model.training = False
        dataset = DataInput(test_set, u_read_list, u_friend_list, uf_read_list, i_read_list, i_friend_list,if_read_list, i_link_list, args.test_batch_size, args.trunc_len, ui_p,set(range(item_count)), False)
        with torch.no_grad():
            ret = rec_test3(model, dataset)
            t3 = time()
            perf_str = 'Rec: Recall=[%.5f,%.5f,%.5f], NDCG=[%.5f,%.5f,%.5f] Time:[%0.2f]' % (ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], t3-t2)
            log.record(perf_str)
            if ret['recall'][0] > best_recall:
                best_recall, best_ndcg, best_iter = ret['recall'][0], ret['ndcg'][0], epoch
    log.record('Best Epoch[%d] Rec: Recall10=[%.5f], NDCG10=[%.5f]' % (best_iter, best_recall, best_ndcg))

