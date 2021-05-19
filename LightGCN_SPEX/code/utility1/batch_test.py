import heapq
import numpy as np
import torch
import utility1.metrics as metrics
from lg_parser import parse_args_r
args = parse_args_r()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            re = test_one_user(u, test_item, neg_item, model)
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
    return result


def test_one_user(user,test_item,neg_item,model):
    u = user
    user_pos_test = test_item
    test_items = neg_item + user_pos_test
    users = np.full(len(test_items), u)
    predictions = model(users=torch.from_numpy(users).long(), items=torch.from_numpy(np.array(test_items)).long(), labels=None, flag=1)
    predictions = predictions.cpu().tolist()
    rating = {}
    for i in range(len(test_items)):
        item = test_items[i]
        rating[item] = predictions[i]
    r = ranklist_by_heapq(user_pos_test, rating)
    return get_performance(user_pos_test, r)


def rec_test(model, testRatings, testNegatives): 
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    user = np.array(list(testRatings.keys()))
    n_test_users = len(user)  

    with torch.no_grad():
        for u in user:
            test_item = testRatings[u]
            neg_item = testNegatives[u]
            re = test_one_user_rec(u, test_item, neg_item, model)
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
    return result

def test_one_user_rec(user,test_item,neg_item,model): 
    u = user
    user_pos_test = test_item
    test_items = neg_item + user_pos_test
    users = np.full(len(test_items), u)
    predictions = model(users=torch.from_numpy(users).long(), items=torch.from_numpy(np.array(test_items)).long(), labels=None, slice_indices=None, trust_data=None, flag=1)
    predictions = predictions.cpu().tolist()
    rating = {}
    for i in range(len(test_items)):
        item = test_items[i]
        rating[item] = predictions[i]
    r = ranklist_by_heapq(user_pos_test, rating)
    return get_performance(user_pos_test, r)

def get_performance(user_pos_test, r):
    recall, ndcg = [], []

    for K in Ks:
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
    return {'recall': np.array(recall),'ndcg': np.array(ndcg)}

def ranklist_by_heapq(user_pos_test, rating):
    K_max = max(Ks)  
    K_max_item_score = heapq.nlargest(K_max, rating, key=rating.get) 

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r



