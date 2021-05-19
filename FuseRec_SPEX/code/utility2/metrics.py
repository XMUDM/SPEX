import numpy as np
import torch
Ks = [10,20,50]

def trust_test(model, test_data):
    l = test_data.length
    recall10, recall20, recall50, ndcg10, ndcg20, ndcg50 = 0, 0, 0, 0, 0, 0
    slices = test_data.generate_batch(256)

    with torch.no_grad():
        for slice_indices in slices:
            trust_scores, trust_targets = model(slice_indices=slice_indices, data=test_data,flag=2)
            for score, target in zip(trust_scores, trust_targets):
                si = score[target].topk(50)[1].cpu().detach().numpy()
                re = test_one_user([len(target)-1], si)
                recall10 += re['recall'][0]
                recall20 += re['recall'][1]
                recall50 += re['recall'][2]
                ndcg10 += re['ndcg'][0]
                ndcg20 += re['ndcg'][1]
                ndcg50 += re['ndcg'][2]
    return [recall10 / l, recall20 / l, recall50 / l, ndcg10 / l, ndcg20 / l, ndcg50 / l]

def trust_test5(model, test_data):
    l = test_data.length
    recall10, recall20, recall50, ndcg10, ndcg20, ndcg50 = 0, 0, 0, 0, 0, 0
    slices = test_data.generate_batch(256)

    with torch.no_grad():
        for slice_indices in slices:
            trust_scores, trust_targets = model(users=None, items=None, labels=None, u_items=None, u_items_mask=None, u_frids=None, u_frids_mask=None, u_frids_items=None, F_i=None, slice_indices=slice_indices, data=test_data,flag=2)
            for score, target in zip(trust_scores, trust_targets):
                si = score[target].topk(50)[1].cpu().detach().numpy()
                re = test_one_user([len(target)-1], si)
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
    precision, recall, ndcg, hit_ratio = [], [], [], []
    for K in Ks:
        recall.append(recall_at_k(r, K, len(pos_item)))
        ndcg.append(ndcg_at_k(r, K))
    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}
            

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


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


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.









