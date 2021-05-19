import os
from fuse_parser import parse_args
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
filename ='log/epinion2_auto2_0.001_nos_p.log'
print(filename)
import random
import numpy as np
import torch
import pickle
from time import time
from collections import defaultdict
from torch.utils.data import DataLoader
from utility.Input import Dataset
from utility.Model_multi import FuseRec_SPEX
from utility.metrics import rec_test2
from utility.Logging import Logging
from utility2.metrics import trust_test5
from utility2.utils import Data

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(args.seed)

log_dir = os.path.join(os.getcwd(), 'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(os.getcwd(), filename)
log = Logging(log_path)


# load data
dataset = Dataset("epinion2")
rec_train_loader = DataLoader(dataset=dataset.generate_train_data(), batch_size=args.batchSize,shuffle=True)
rec_test_loader = DataLoader(dataset=dataset.generate_test_data(), batch_size=100, shuffle=False)
# load data
trust_train_data = pickle.load(open('../data/epinion2/trust/train.txt', 'rb'))
trust_test_data = pickle.load(open('../data/epinion2/trust/test2.txt', 'rb'))
trust_train_loader = Data(trust_train_data, dataset.user_num, shuffle=False)
trust_test_loader = Data(trust_test_data, dataset.user_num, shuffle=False, test=True)
user_path_indx = defaultdict(list)
path = trust_train_data[0]
for i, p in zip(range(len(path)), path):
    u = p[0]
    user_path_indx[u].append(i)
all_path_index = set(range(len(path)))
trust_batch_size = len(all_path_index) // (dataset.train_len*6//256)
print("trust_batch_size:", trust_batch_size)



if __name__ == "__main__":
    model = FuseRec_SPEX(dataset.user_num, dataset.item_num, args.hiddenSize, dataset.load_type()).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  

    bestres = [0, 0, 0, 0, 0, 0]
    besttru = [0, 0, 0, 0, 0, 0]
    for epoch in range(args.epochs):
        # train
        t1 = time()
        model.train()
        total_loss1,total_loss2 = 0.0, 0.0
        for data in rec_train_loader:
            optimizer.zero_grad()

            user, item, label, u_items, u_items_mask, u_frids, u_frids_mask, u_frids_items, F_i = data
            unique_user = set(user.numpy().tolist())
            path_index = []
            for u in unique_user:
                path_index.extend(user_path_indx[u])
            if len(path_index) > trust_batch_size:
                path_index = random.sample(path_index, trust_batch_size)
            if len(path_index) < trust_batch_size:  # avoid this batch users do not have path
                path_index = random.sample(all_path_index, trust_batch_size)
            slice_indices = np.array(list(path_index), dtype=int)

            loss1,loss2 = model(user.cuda(), item.cuda(), label.cuda(), u_items.cuda(), u_items_mask.cuda(), u_frids.cuda(), u_frids_mask.cuda(), u_frids_items.cuda(), F_i.cuda(),slice_indices,trust_train_loader, 0)
            reg_loss = model.reg_loss()
            # multi_task
            T = len(path_index)
            n_rec = 5
            T_rec = len(user)
            precision1 = torch.exp(-2 * model.task_weights[0])
            precision2 = torch.exp(-2 * model.task_weights[1])
            loss = precision1 * (loss1 + 0.001*reg_loss) + precision2 * loss2 + 2 * (n_rec + 1) * T_rec * model.task_weights[0] + T * model.task_weights[1]
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

        t2 = time()
        train_rec = 'Epoch [%d], P1 [%.5f], P2 [%.5f], Loss1 [%.5f], Loss2 [%.5f], Time[%.4f]' % (epoch, precision1, precision2,total_loss1, total_loss2, t2-t1)
        log.record(train_rec)

        # test
        model.eval()
        ret = rec_test2(model, rec_test_loader, dataset.test_len / 100)
        t3 = time()
        perf_str = 'Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f], Time=[%.4f]' % (
        epoch, ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], t3 - t2)
        log.record(perf_str)
        for i in range(len(bestres)):
            bestres[i] = max(bestres[i], ret[i])

        # test
        ret2 = trust_test5(model, trust_test_loader)
        t4 = time()
        perf_str = 'Tru:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f], Time=[%.4f]' % (epoch, ret2[0], ret2[1], ret2[2], ret2[3], ret2[4], ret2[5], t4-t3)
        log.record(perf_str)
        for i in range(len(besttru)):
            besttru[i] = max(besttru[i], ret2[i])


    perf_str = 'RecBest: recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (bestres[0], bestres[1], bestres[2], bestres[3], bestres[4], bestres[5])
    log.record(perf_str)
    perf_str = 'TruBest: recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (besttru[0], besttru[1], besttru[2], besttru[3], besttru[4], besttru[5])
    log.record(perf_str)

