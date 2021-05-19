import numpy as np
Ks = [10,20,50]

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

def rec_test(model, d_test):
    scores = np.reshape(model(d_test.test_user_list, d_test.test_item_list, None, None, None, 1), [-1, 100])
    result_rec = {'recall': np.zeros(len(Ks)),'ndcg': np.zeros(len(Ks))}
    index = np.argsort(-scores)
    index = np.int64(index <= 0)

    for r in index:
        re = get_performance(r)
        result_rec['recall'] += re['recall']
        result_rec['ndcg'] += re['ndcg']

    l = len(index)
    result_rec['recall'] = result_rec['recall']/l
    result_rec['ndcg'] = result_rec['ndcg']/l
    return result_rec

def rec_test_single(model, d_test):
    scores = np.reshape(model(d_test.test_user_list, d_test.test_item_list, None, 1), [-1, 100])
    result_rec = {'recall': np.zeros(len(Ks)),'ndcg': np.zeros(len(Ks))}
    index = np.argsort(-scores)
    index = np.int64(index <= 0)

    for r in index:
        re = get_performance(r)
        result_rec['recall'] += re['recall']
        result_rec['ndcg'] += re['ndcg']

    l = len(index)
    result_rec['recall'] = result_rec['recall']/l
    result_rec['ndcg'] = result_rec['ndcg']/l
    return result_rec

def get_performance(r):
    recall, ndcg = [], []
    for K in Ks:
        recall.append(recall_at_k(r, K, 1))
        ndcg.append(ndcg_at_k(r, K))
    return {'recall': np.array(recall),'ndcg': np.array(ndcg)}
