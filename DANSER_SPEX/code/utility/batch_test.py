import torch
import numpy as np
from parser_a import parse_args

args = parse_args()
Ks = args.Ks

def to_gpuTensor(l):
    return torch.tensor(l).long().cuda()

# multi_task
def rec_test(model, dataset):
    result_rec = {'recall': np.zeros(len(args.Ks)), 'ndcg': np.zeros(len(args.Ks))}
    with torch.no_grad():
        l_test = dataset.epoch_size
        for users, items, labels, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_linput, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput in dataset:
            predictions = model(to_gpuTensor(users), to_gpuTensor(items), to_gpuTensor(labels),
                                to_gpuTensor(u_readinput),to_gpuTensor(u_friendinput),
                                to_gpuTensor(uf_readinput), to_gpuTensor(i_readinput),
                                to_gpuTensor(i_friendinput),
                                to_gpuTensor(if_readinput), None, None, 1)
            index = np.argsort(-predictions.cpu())
            r = np.int64(index <= 0)

            re = get_performance(r)
            result_rec['recall'] += re['recall']
            result_rec['ndcg'] += re['ndcg']

        result_rec['recall'] = result_rec['recall'] / l_test
        result_rec['ndcg'] = result_rec['ndcg'] / l_test
    return result_rec

# multi_task
def rec_test1(model, dataset):
    result_rec = {'recall': np.zeros(len(args.Ks)), 'ndcg': np.zeros(len(args.Ks))}
    with torch.no_grad():
        l_test = dataset.epoch_size
        for u, i, l, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_linput, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput in dataset:
            predictions = model(to_gpuTensor(u), to_gpuTensor(i), to_gpuTensor(l), to_gpuTensor(u_readinput),
                  to_gpuTensor(u_friendinput), to_gpuTensor(uf_readinput), to_gpuTensor(u_read_l),
                  to_gpuTensor(u_friend_l), to_gpuTensor(uf_read_linput), to_gpuTensor(i_readinput),
                  to_gpuTensor(i_friendinput), to_gpuTensor(if_readinput), to_gpuTensor(i_linkinput),
                  to_gpuTensor(i_read_l), to_gpuTensor(i_friend_l), to_gpuTensor(if_read_linput), None, None, 1)
            index = np.argsort(-predictions.cpu())
            r = np.int64(index <= 0)

            re = get_performance(r)
            result_rec['recall'] += re['recall']
            result_rec['ndcg'] += re['ndcg']

        result_rec['recall'] = result_rec['recall'] / l_test
        result_rec['ndcg'] = result_rec['ndcg'] / l_test
    return result_rec
# rec_task
def rec_test2(model, dataset):
    result_rec = {'recall': np.zeros(len(args.Ks)), 'ndcg': np.zeros(len(args.Ks))}
    with torch.no_grad():
        for users, items, labels, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_linput, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput in dataset:

            predictions = model(to_gpuTensor(users), to_gpuTensor(items), to_gpuTensor(labels),
                                to_gpuTensor(u_readinput),to_gpuTensor(u_friendinput),
                                to_gpuTensor(uf_readinput), to_gpuTensor(i_readinput),
                                to_gpuTensor(i_friendinput),
                                to_gpuTensor(if_readinput), 0)
            index = np.argsort(-predictions.cpu())
            r = np.int64(index <= 0)
            re = get_performance(r)
            result_rec['recall'] += re['recall']
            result_rec['ndcg'] += re['ndcg']
        l_test = dataset.epoch_size
        result_rec['recall'] = result_rec['recall'] / l_test
        result_rec['ndcg'] = result_rec['ndcg'] / l_test
    return result_rec

# rec_task
def rec_test3(model, dataset):
    result_rec = {'recall': np.zeros(len(args.Ks)), 'ndcg': np.zeros(len(args.Ks)),}
    with torch.no_grad():
        for u, i, l, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_linput, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput in dataset:
            predictions = model(to_gpuTensor(u), to_gpuTensor(i), to_gpuTensor(l), to_gpuTensor(u_readinput),
                  to_gpuTensor(u_friendinput), to_gpuTensor(uf_readinput), to_gpuTensor(u_read_l),
                  to_gpuTensor(u_friend_l), to_gpuTensor(uf_read_linput), to_gpuTensor(i_readinput),
                  to_gpuTensor(i_friendinput), to_gpuTensor(if_readinput), to_gpuTensor(i_linkinput),
                  to_gpuTensor(i_read_l), to_gpuTensor(i_friend_l), to_gpuTensor(if_read_linput), 0)
            index = np.argsort(-predictions.cpu())
            r = np.int64(index <= 0)
            re = get_performance(r)
            result_rec['recall'] += re['recall']
            result_rec['ndcg'] += re['ndcg']
        l_test = dataset.epoch_size
        result_rec['recall'] = result_rec['recall'] / l_test
        result_rec['ndcg'] = result_rec['ndcg'] / l_test
    return result_rec


def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    if all_pos_num == 0:
        result = 0.0
    else:
        result = np.sum(r) / all_pos_num
    return result



def get_performance(r):
    recall, ndcg = [], []
    for K in Ks:
        recall.append(recall_at_k(r, K, 1))
        ndcg.append(ndcg_at_k(r, K))
    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}
