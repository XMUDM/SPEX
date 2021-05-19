import torch
import numpy as np
import heapq
import utility1.metrics as metrics
Ks = [10,20,50]


def test_rec(model, test_item, neg_item, n_users, max_friend, tfset):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
    user = test_item.keys()
    n_test_users = len(user)

    with torch.no_grad():
        for u in user:
            input_user = [u] * 100
            input_item = test_item[u]
            input_neg_item = neg_item[u]
            input_uf = []
            one_use = n_users * np.ones(max_friend, np.int)
            if u in tfset:
                one_use = tfset[u]
            for i in range(100):
                input_uf.append(one_use)
            re = test_one_rec_user(input_user, input_item, input_neg_item, input_uf, model)
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
    return result

def test_one_rec_user(input_user,input_item,input_neg_item,input_uf,model):
    user_pos_test = input_item
    test_items = input_neg_item + user_pos_test
    predictions = model(input_u=torch.LongTensor(input_user).cuda(),input_i=torch.LongTensor(test_items).cuda(), label=None, input_uf=torch.LongTensor(input_uf).cuda(),i=None,data=None, flag=1)
    predictions = predictions.cpu().tolist()
    score = {}
    for i in range(len(test_items)):
        item = test_items[i]
        score[item] = predictions[i]
    r = ranklist_by_heapq(user_pos_test, score)
    return get_performance(user_pos_test, r)


def test_trust(model, test_data):
    l = test_data.length
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
    slices = test_data.generate_batch(model.batch_size)

    with torch.no_grad():
        for i in slices:
            trust_scores, trust_targets = model(None, None, None, None, i, test_data, 2)
            sub_indexs = trust_scores.topk(50)[1]
            sub_indexs = sub_indexs.cpu().detach().numpy()
            for index, target in zip(sub_indexs, trust_targets.cpu().detach().numpy()):
                re = test_one_user([target], index)
                result['recall'][0] += re['recall'][0]
                result['recall'][1] += re['recall'][1]
                result['recall'][2] += re['recall'][2]
                result['ndcg'][0] += re['ndcg'][0]
                result['ndcg'][1] += re['ndcg'][1]
                result['ndcg'][2] += re['ndcg'][2]
    result['recall'] /= l
    result['ndcg'] /= l
    return result

def test_trust5(model, test_data):
    l = test_data.length
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}
    slices = test_data.generate_batch(model.batch_size)

    with torch.no_grad():
        for i in slices:
            trust_scores, trust_targets = model(None, None, None, None, i, test_data, 2)
            for score, target in zip(trust_scores, trust_targets):
                si = score[target].topk(50)[1].cpu().detach().numpy()
                re = test_one_user([len(target) - 1], si)
                result['recall'][0] += re['recall'][0]
                result['recall'][1] += re['recall'][1]
                result['recall'][2] += re['recall'][2]
                result['ndcg'][0] += re['ndcg'][0]
                result['ndcg'][1] += re['ndcg'][1]
                result['ndcg'][2] += re['ndcg'][2]
    result['recall'] /= l
    result['ndcg'] /= l
    return result

def test_one_user(pos_item, K_max_item):
    target = np.array(pos_item * 50)
    d = K_max_item - target
    r = np.int32(d == 0)
    return get_performance(pos_item, r)

def get_performance(user_pos_test, r):
    recall, ndcg = [], []

    for K in Ks:
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}


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



