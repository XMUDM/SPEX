import torch
import utility.metrics as metrics
import multiprocessing
import heapq
import numpy as np
from utility.gpuutil import trans_to_cpu, trans_to_cuda

cores = multiprocessing.cpu_count() // 2
Ks = [10,20,50]
BATCH_SIZE = 256

def test(model, testRatings, testNegatives):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    user = np.array(list(testRatings.keys()))
    n_test_users = len(user)

    with torch.no_grad():
        for u in user:
            test_item = testRatings[u]
            neg_item = testNegatives[u]
            re = test_one_user2(u,test_item,neg_item,model)
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users

    return result

def rec_test(model, testRatings, testNegatives):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    user = np.array(list(testRatings.keys()))
    n_test_users = len(user)

    with torch.no_grad():
        for u in user:
            test_item = testRatings[u]
            neg_item = testNegatives[u]
            re = test_one_user(u,test_item,neg_item,model)
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users

    return result


def test_one_user(user,test_item,neg_item,model):
    u = user
    user_pos_test = test_item
    test_items = neg_item + user_pos_test
    users = np.full(len(test_items), u)
    predictions = model(user=trans_to_cuda(torch.from_numpy(users).long()),
                        item=trans_to_cuda(torch.from_numpy(np.array(test_items)).long()), slice_indices=None, trust_data=None, flag=1)
    predictions = predictions.cpu().tolist()
    rating = {}
    for i in range(len(test_items)):
        item = test_items[i]
        rating[item] = predictions[i]
    r, auc = ranklist_by_heapq(user_pos_test, rating)
    return get_performance(user_pos_test, r)

def test_one_user2(user,test_item,neg_item,model):
    u = user
    user_pos_test = test_item
    test_items = neg_item + user_pos_test
    users = np.full(len(test_items), u)
    predictions = model(user=trans_to_cuda(torch.from_numpy(users).long()),
                        item=trans_to_cuda(torch.from_numpy(np.array(test_items)).long()))
    predictions = predictions.cpu().tolist()
    rating = {}
    for i in range(len(test_items)):
        item = test_items[i]
        rating[item] = predictions[i]
    r, auc = ranklist_by_heapq(user_pos_test, rating)
    return get_performance(user_pos_test, r)

def get_performance(user_pos_test, r):
    recall, ndcg = [], []

    for K in Ks:
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}

def ranklist_by_heapq(user_pos_test, rating):
    K_max = max(Ks)  # lly
    K_max_item_score = heapq.nlargest(K_max, rating, key=rating.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc  # [1,0,0,0,1...]

def ranklist_by_sorted(user_pos_test, rating):
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, rating, key=rating.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(rating, user_pos_test)
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
