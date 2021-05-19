import os
from parser import parse_args

args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id

import pickle
import random
from collections import defaultdict
import numpy as np
import torch.utils.data

from utility1.Logging import Logging
from utility1.load_data import Data as D1, get_train_instances
from utility1.batch_test import test_rec, test_trust5
from utility1.Model_expert_s import SAMN_TRUST
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
log_path = os.path.join(os.getcwd(), 'log/%s_11.log' % (args.dataset))
log = Logging(log_path)


def train(model, optimizer, train_loader, epoch):
    trust_batch_size = len(alluser_path_index) // len(train_loader)
    t_loss1, t_loss2 = 0.0, 0.0
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
        loss = loss1+loss2
        loss.backward()
        optimizer.step()

        t_loss1 += loss1.item()
        t_loss2 += loss2.item()
    log.record('%d,%.5f,%.5f' % (epoch, t_loss1, t_loss2))  # 画图文件


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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
