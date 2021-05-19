from utility1 import metrics as metrics
import heapq
import numpy as np
Ks = [10,20,50]

def test_one_user(pos_item, all_items, rating):
    r, auc = ranklist_by_heapq(pos_item, all_items, rating, Ks)
    return get_performance(pos_item, r)

def test_one_user2(pos_item, K_max_item):
    target = np.array(pos_item * 50)
    d = K_max_item - target
    r = np.int32(d == 0)
    return get_performance(pos_item, r)

def ranklist_by_heapq(pos_item, all_items, rating, Ks):
    path_score = {}
    for i in range(len(all_items)):
        path_score[all_items[i]] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, path_score, key=path_score.get)

    r = []
    for i in K_max_item_score:
        if i in pos_item:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc


def get_performance(pos_item, r):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        recall.append(metrics.recall_at_k(r, K, len(pos_item)))
        ndcg.append(metrics.ndcg_at_k(r, K))
    return {'recall': np.array(recall),'ndcg': np.array(ndcg)}