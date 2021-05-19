import numpy as np
import torch
Ks = [10, 20, 50]

def rec_test(model, test_loader, l):
    recall10, recall20, recall50, ndcg10, ndcg20, ndcg50 = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for data in test_loader:
            user, item, label, u_items, u_items_mask, u_frids, u_frids_mask, u_frids_items, F_i = data
            scores = model(user.cuda(), item.cuda(), label.cuda(), u_items.cuda(), u_items_mask.cuda(), u_frids.cuda(), u_frids_mask.cuda(), u_frids_items.cuda(), F_i.cuda(), 1)
            sub_indexs = scores.topk(50)[1]
            sub_indexs = sub_indexs.detach().cpu().numpy()
            re = test_one_user([99], sub_indexs)
            recall10 += re['recall'][0]
            recall20 += re['recall'][1]
            recall50 += re['recall'][2]
            ndcg10 += re['ndcg'][0]
            ndcg20 += re['ndcg'][1]
            ndcg50 += re['ndcg'][2]
    return [recall10 / l, recall20 / l, recall50 / l, ndcg10 / l, ndcg20 / l, ndcg50 / l]


def rec_test2(model, test_loader, l):
    recall10, recall20, recall50, ndcg10, ndcg20, ndcg50 = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for data in test_loader:
            user, item, label, u_items, u_items_mask, u_frids, u_frids_mask, u_frids_items, F_i = data
            scores = model(user.cuda(), item.cuda(), label.cuda(), u_items.cuda(), u_items_mask.cuda(), u_frids.cuda(),u_frids_mask.cuda(), u_frids_items.cuda(), F_i.cuda(),None, None, 1)
            sub_indexs = scores.topk(50)[1]
            sub_indexs = sub_indexs.detach().cpu().numpy()
            re = test_one_user([99], sub_indexs)
            recall10 += re['recall'][0]
            recall20 += re['recall'][1]
            recall50 += re['recall'][2]
            ndcg10 += re['ndcg'][0]
            ndcg20 += re['ndcg'][1]
            ndcg50 += re['ndcg'][2]
    return [recall10 / l, recall20 / l, recall50 / l, ndcg10 / l, ndcg20 / l, ndcg50 / l]


def test_one_user(pos_item, K_max_item):
    target = np.array(pos_item * 50)
    d = K_max_item - target
    r = np.int32(d == 0)
    return get_performance(pos_item, r)


def get_performance(pos_item, r):
    recall, ndcg = [], []

    for K in Ks:
        recall.append(recall_at_k(r, K, len(pos_item)))
        ndcg.append(ndcg_at_k(r, K))
    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}


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
