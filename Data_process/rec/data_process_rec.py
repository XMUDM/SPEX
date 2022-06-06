import os
import argparse
import pickle
import random
import shutil
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='epinion2', help='epinion2/weibo/twitter')

    return parser.parse_args()


def change_format():
    '''
    change file format
    '''
    m_rating = loadmat(os.path.join(args.root, "rating_with_timestamp.mat"))
    m_trust = loadmat(os.path.join(args.root, "trust_with_timestamp.mat"))
    # print(m_trust)
    # print(m_rating)
    df_rating = pd.DataFrame(m_rating['rating'])
    # df_rating = pd.DataFrame(m_rating['rating_with_timestamp'])
    df_trust = pd.DataFrame(m_trust['trust'])
    df_rating.to_csv(os.path.join(args.root, "rating_1.txt"), header=None, index=False, sep=" ")
    df_trust.to_csv(os.path.join(args.root, "trust_1.txt"), header=None, index=False, sep=" ")
    print("change_format")


def select_items():
    '''
    Recommend: Select users with more than 10 interactions
    '''
    u_lists = defaultdict(list)
    i_lists = defaultdict(list)
    c_lists = defaultdict(list)
    r_lists = defaultdict(list)
    t_lists = defaultdict(list)
    with open(os.path.join(args.root, 'rating_1.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user,product,category,rating,helpfulness,time
            u = line_arr[0]
            i = line_arr[1]
            c = line_arr[2]
            r = line_arr[3]
            t = line_arr[5]
            if i not in set(i_lists[u]):
                i_lists[u].append(i)
                r_lists[u].append(r)
                c_lists[u].append(c)
                t_lists[u].append(t)
                u_lists[i].append(u)

    with open(os.path.join(args.root, 'rating_2.txt'), 'w') as f:
        for u in i_lists:
            for i in range(len(i_lists[u])):
                if len(u_lists[i_lists[u][i]]) > 10:
                    f.write(u + " " + i_lists[u][i] + " " + c_lists[u][i] + " " + r_lists[u][i] + " " + t_lists[u][
                        i] + "\n")
    print("select_items")


def select_users():
    '''
    Duplicate data removal: the same user reviews the same product multiple times
    Recommendation: Select users with more than 10 interactions
    Trust: Select users with a history of reviews
    '''
    i_lists = defaultdict(list)
    c_lists = defaultdict(list)
    r_lists = defaultdict(list)
    t_lists = defaultdict(list)
    with open(os.path.join(args.root, 'rating_2.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user,product,rating,time
            u = line_arr[0]
            i = line_arr[1]
            c = line_arr[2]
            r = line_arr[3]
            t = line_arr[4]
            if i not in set(i_lists[u]):
                i_lists[u].append(i)
                c_lists[u].append(c)
                r_lists[u].append(r)
                t_lists[u].append(t)

    u_set = set()
    with open(os.path.join(args.root, 'rating_3.txt'), 'w') as f:
        for u in i_lists:
            l = len(i_lists[u])
            if l >= 30:
                u_set.add(u)
                for i in range(l):
                    f.write(u + " " + i_lists[u][i] + " " + c_lists[u][i] + " " + r_lists[u][i] + " " + t_lists[u][
                        i] + "\n")

    u1_list, u2_list, tt_list = [], [], []
    with open(os.path.join(args.root, 'trust_1.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user1,user2,time
            u1 = line_arr[0]
            u2 = line_arr[1]
            t = line_arr[2]
            if (u1 in u_set) and (u2 in u_set):
                u1_list.append(u1)
                u2_list.append(u2)
                tt_list.append(t)

    with open(os.path.join(args.root, 'trust_2.txt'), 'w') as f:
        l = len(u1_list)
        for i in range(l):
            f.write(u1_list[i] + " " + u2_list[i] + " " + tt_list[i] + "\n")
    print("select_users")


def unify_index():
    '''
    Reindex user and item
    '''
    u_list, i_list, c_list, r_list, t_list = [], [], [], [], []
    with open(os.path.join(args.root, 'rating_3.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user,product,rating,time
            u = line_arr[0]
            i = line_arr[1]
            c = line_arr[2]
            r = line_arr[3]
            t = line_arr[4]
            u_list.append(u)
            i_list.append(i)
            c_list.append(c)
            r_list.append(r)
            t_list.append(t)

    u_set = set(u_list)
    i_set = set(i_list)
    c_set = set(c_list)
    d_u = dict(zip(u_set, range(len(u_set))))
    d_i = dict(zip(i_set, range(len(i_set))))
    d_c = dict(zip(c_set, range(len(c_set))))

    with open(os.path.join(args.root, 'rating_4.txt'), 'w') as f:
        for i in range(len(u_list)):
            f.write(
                str(d_u[u_list[i]]) + " " + str(d_i[i_list[i]]) + " " + str(d_c[c_list[i]]) + " " + r_list[i] + " " +
                t_list[i] + "\n")

    u1_list, u2_list, tt_list = [], [], []
    with open(os.path.join(args.root, 'trust_2.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user1,user2,time
            u1 = line_arr[0]
            u2 = line_arr[1]
            t = line_arr[2]
            u1_list.append(u1)
            u2_list.append(u2)
            tt_list.append(t)

    with open(os.path.join(args.root, 'trust_3.txt'), 'w') as f:
        for i in range(len(u1_list)):
            f.write(str(d_u[u1_list[i]]) + " " + str(d_u[u2_list[i]]) + " " + tt_list[i] + "\n")
    print("unify_index")


def sort():
    '''
    Sort: for check
    '''
    df_rating = pd.read_csv(os.path.join(args.root, 'rating_4.txt'), sep="\s+",
                            names=['uid', 'iid', 'type', 'rating', 'time'],
                            usecols=[0, 1, 2, 3, 4])
    df_trust = pd.read_csv(os.path.join(args.root, 'trust_3.txt'), sep="\s+", names=['uid1', 'uid2', 'time'],
                           usecols=[0, 1, 2])
    df_rating.sort_values(by=['uid', 'time'], inplace=True)
    df_trust.sort_values(by=['uid1', 'time'], inplace=True)
    df_rating.to_csv(os.path.join(args.root, 'rating_5.txt'), header=None, index=False, sep=" ")
    df_trust.to_csv(os.path.join(args.root, 'trust_4.txt'), header=None, index=False, sep=" ")

    '''
    Save all items and their corresponding types
    '''
    item = list(df_rating['iid'])
    type = list(df_rating['type'])

    unique_i = []
    i_type = []
    for i, c in zip(item, type):
        if i not in unique_i:
            unique_i.append(i)
            i_type.append(c)

    unique_i_c = pd.DataFrame({'i': unique_i, 'c': i_type})
    unique_i_c.sort_values(by='i', inplace=True)
    unique_i_c.to_csv(os.path.join(args.root, 'type.txt'), header=None, index=False, sep=" ")
    print("sort")


def split():
    '''
    The retention method is used to divide training set and test set
    Negative sampling for the test set
    Save all_u_list,all_i_list,all_social_list
    '''
    all_u_list, all_i_list, all_social_list = defaultdict(list), defaultdict(list), defaultdict(list)

    ui_list, ur_list, ut_list = defaultdict(list), defaultdict(list), defaultdict(list)
    with open(os.path.join(args.root, 'rating_5.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user,product,rating,time
            u = line_arr[0]
            i = line_arr[1]
            r = line_arr[2]
            t = line_arr[3]
            ui_list[u].append(i)
            ur_list[u].append(r)
            ut_list[u].append(t)

            all_u_list[u].append(i)
            all_i_list[i].append(u)
            all_social_list[u].append(u)

    u1u2_list, u1t_list = defaultdict(list), defaultdict(list)
    with open(os.path.join(args.root, 'trust_4.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user1,user2,time
            u1 = line_arr[0]
            u2 = line_arr[1]
            t = line_arr[2]
            u1u2_list[u1].append(u2)
            u1t_list[u1].append(t)
            all_social_list[u1].append(u2)

    with open(os.path.join(args.root, 'train_rating.txt'), 'w') as f:
        for u in ui_list:
            for i in range(len(ui_list[u]) - 1):
                f.write(u + " " + ui_list[u][i] + " " + ur_list[u][i] + " " + ut_list[u][i] + "\n")

    all_item = set(all_i_list.keys())
    with open(os.path.join(args.root, 'test_rating.txt'), 'w') as f:
        for u in ui_list:
            opt_items = all_item - set(ui_list[u])
            neg_items = random.sample(opt_items, 99)
            for i in range(99):
                f.write(u + " " + neg_items[
                    i] + " " + "0" + " " + "0" + "\n")  # neg_score:0 neg_time:0 (pos_score>0 pos_time>0)
            i = len(ui_list[u]) - 1
            f.write(u + " " + ui_list[u][i] + " " + ur_list[u][i] + " " + ut_list[u][i] + "\n")

    with open(os.path.join(args.root, 'train_trust.txt'), 'w') as f:
        for u in u1u2_list:
            if len(u1u2_list[u]) == 1:
                f.write(u + " " + u1u2_list[u][0] + " " + u1t_list[u][0] + "\n")
            elif len(u1u2_list[u]) > 1:
                for i in range(len(u1u2_list[u]) - 1):
                    f.write(u + " " + u1u2_list[u][i] + " " + u1t_list[u][i] + "\n")

    all_user = set(all_u_list.keys())
    with open(os.path.join(args.root, 'test_trust.txt'), 'w') as f:
        for u in u1u2_list:
            l = len(u1u2_list[u])
            if l > 1:
                opt_users = all_user - set(u1u2_list[u])
                neg_users = random.sample(opt_users, 99)
                for i in range(99):
                    f.write(u + " " + neg_users[i] + " " + "0" + "\n")  # neg_time:0 (pos_time>0)
                f.write(u + " " + u1u2_list[u][l - 1] + " " + u1t_list[u][l - 1] + "\n")
    print("split")


def to_NGCF():
    train_uv_list = defaultdict(list)
    with open(os.path.join(args.root, 'train_rating.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user,product,rating,time
            u = line_arr[0]
            v = line_arr[1]
            train_uv_list[u].append(v)

    Path(os.path.join("..", "..", "NGCF_SPEX", "data", args.root, "rec")).mkdir(parents=True, exist_ok=True)
    with open(os.path.join("..", "..", "NGCF_SPEX", "data", args.root, "rec", 'train.txt'), 'w') as f:
        for u in train_uv_list:
            f.write(u)
            for v in train_uv_list[u]:
                f.write(" " + v)
            f.write("\n")

    test_u_p, test_v_p = [], []
    test_uv_n = defaultdict(list)
    with open(os.path.join(args.root, 'test_rating.txt'), 'r') as f:
        i = 1
        for line in f:
            line_arr = line.strip().split()  # user,product,rating,time
            u = line_arr[0]
            v = line_arr[1]
            if (i % 100) == 0:
                test_u_p.append(u)
                test_v_p.append(v)
            else:
                test_uv_n[u].append(v)
            i += 1

    with open(os.path.join("..", "..", "NGCF_SPEX", "data", args.root, "rec", 'test.txt'), 'w') as f:
        for i in range(len(test_u_p)):
            f.write(test_u_p[i] + " " + test_v_p[i] + "\n")

    with open(os.path.join("..", "..", "NGCF_SPEX", "data", args.root, "rec", 'negative.txt'), 'w') as f:
        for u in test_uv_n:
            f.write(u)
            for v in test_uv_n[u]:
                f.write(" " + v)
            f.write("\n")
    print("to_NGCF")


def to_GraphRec():
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists = defaultdict(list), defaultdict(
        list), defaultdict(list), defaultdict(list)
    social_adj_lists = defaultdict(set)
    ratings_list = {0: 0, 1: 1}

    item = set()
    train_u, train_v, train_r = [], [], []
    with open(os.path.join(args.root, 'train_rating.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user,product,rating,time
            u = int(line_arr[0])
            v = int(line_arr[1])
            r = line_arr[2]
            t = line_arr[3]
            train_u.append(u)
            train_v.append(v)
            train_r.append(1)
            item.add(v)

            history_u_lists[u].append(v)
            history_ur_lists[u].append(1)
            history_v_lists[v].append(u)
            history_vr_lists[v].append(1)

            social_adj_lists[u].add(u)

    train_u1, train_u2, train_tr = [], [], []
    with open(os.path.join(args.root, 'train_trust.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user1,user2,time
            u1 = int(line_arr[0])
            u2 = int(line_arr[1])
            t = line_arr[2]
            train_u1.append(u1)
            train_u2.append(u2)
            train_tr.append(1)

            social_adj_lists[u1].add(u2)

    test_u, test_v, test_r = [], [], []
    with open(os.path.join(args.root, 'test_rating.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user,product,rating,time
            u = int(line_arr[0])
            v = int(line_arr[1])
            r = int(line_arr[2])
            test_u.append(u)
            test_v.append(v)
            if r > 0:
                test_r.append(1)
                item.add(v)
            else:
                test_r.append(0)
                item.add(v)

    test_u1, test_u2, test_tr = [], [], []
    with open(os.path.join(args.root, 'test_trust.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user,user2,time
            u1 = int(line_arr[0])
            u2 = int(line_arr[1])
            t = int(line_arr[2])
            test_u1.append(u1)
            test_u2.append(u2)
            if t > 0:
                test_tr.append(1)
            else:
                test_tr.append(0)

    Path(os.path.join("..", "..", "GraphRec_SPEX", "data", args.root, "rec")).mkdir(parents=True, exist_ok=True)
    with open(os.path.join("..", "..", "GraphRec_SPEX", "data", args.root, "rec", args.root + "_t_dataset.pickle"),
              'wb') as f:
        pickle.dump((history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r,
                     test_u, test_v, test_r,
                     train_u1, train_u2, train_tr, test_u1, test_u2, test_tr, social_adj_lists, ratings_list), f)
    print("to_GraphRec")


def to_NCF():
    df_train = pd.read_csv(os.path.join(args.root, 'train_rating.txt'), sep=" ", names=['uid', 'oid'], usecols=[0, 1])
    df_test = pd.read_csv(os.path.join(args.root, 'test_rating.txt'), sep=" ", names=['uid', 'oid'], usecols=[0, 1])

    df_train['r'] = 1
    df_test['r'] = 1
    Path(os.path.join("..", "..", "NCF_SPEX", "data", args.root, "rec")).mkdir(parents=True, exist_ok=True)
    df_train.to_csv(os.path.join("..", "..", "NCF_SPEX", "data", args.root, "rec", args.root + ".train.rating"),
                    header=None,
                    index=False, sep=str(' '))
    df_test.to_csv(os.path.join("..", "..", "NCF_SPEX", "data", args.root, "rec", args.root + ".test.rating"),
                   header=None,
                   index=False, sep=str(' '))
    shutil.copy(os.path.join("..", "..", "NGCF_SPEX", "data", args.root, "rec", 'negative.txt'),
                os.path.join("..", "..", "NCF_SPEX", "data", args.root, "rec", args.root + ".test.negative"))
    print("to_NCF")


def to_FuseRec():
    u_items, u_frids, F_i = defaultdict(list), defaultdict(list), defaultdict(list)

    item = set()
    train_u, train_v, train_r, train_t = [], [], [], []
    with open(os.path.join(args.root, 'train_rating.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user,product,rating,time
            u = int(line_arr[0])
            v = int(line_arr[1])
            r = line_arr[2]
            t = line_arr[3]
            train_u.append(u)
            train_v.append(v)
            train_r.append(r)

            u_items[u].append(v)
            train_t.append(len(u_items[u]))

    with open(os.path.join(args.root, 'train_trust.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user1,user2,time
            u1 = int(line_arr[0])
            u2 = int(line_arr[1])
            t = line_arr[2]
            u_frids[u1].append(u2)

    test_u, test_v, test_r = [], [], []
    with open(os.path.join(args.root, 'test_rating.txt'), 'r') as f:
        for line in f:
            line_arr = line.strip().split()  # user,product,rating,time
            u = int(line_arr[0])
            v = int(line_arr[1])
            r = int(line_arr[2])
            test_u.append(u)
            test_v.append(v)
            if r > 0:
                test_r.append(1)
                item.add(v)
            else:
                test_r.append(0)
                item.add(v)

    df = pd.read_csv(os.path.join(args.root, 'type.txt'), sep=" ", names=['i', 'c'], usecols=[0, 1])
    i_c = dict(zip(df['i'], df['c']))

    u_num = max(train_u) + 1
    i_num = max(train_v) + 1
    u_i_matrix = coo_matrix((np.array([1] * len(train_u)), (np.array(train_u), np.array(train_v))),
                            shape=(u_num, i_num), dtype=np.int).toarray()
    i_i_matrix = np.matmul(u_i_matrix.transpose(), u_i_matrix)

    fil = []
    for ind in range(i_num):
        d = dict(zip(list(range(i_num)), i_i_matrix[ind]))
        d_order = sorted(d.items(), key=lambda x: x[1], reverse=True)  # [('a', 1), ('b', 2), ('c', 3)]
        f1, f2 = [], []
        for k, v in d_order:
            if len(f1) < 10:
                f1.append(k)
            if (len(f2) < 10) and (i_c[ind] == i_c[k]):
                f2.append(k)
            if (len(f1) == 10) and (len(f2) == 10):
                break
        if len(f2) < 10:
            fil.append(len(f2))
            f2.extend(f1[:(10 - len(f2))])
        F_i[ind].append(f1)
        F_i[ind].append(f2)

    allu = set(train_u)
    for u in allu:
        u_frids[u].append(u)

    # padding
    for i in F_i:
        if len(F_i[i][1]) < 10:
            F_i[i][1].extend([5] * (10 - len(F_i[i][1])))

    Path(os.path.join("..", "..", "FuseRec_SPEX", "data", args.root, "rec")).mkdir(parents=True, exist_ok=True)
    with open(os.path.join("..", "..", "FuseRec_SPEX", "data", args.root, "rec", "relation.pickle"), 'wb') as f:
        pickle.dump((u_items, u_frids, F_i), f)

    train = pd.DataFrame({'u': train_u, 'i': train_v, 't': train_t})
    train.to_csv(os.path.join("..", "..", "FuseRec_SPEX", "data", args.root, "rec", 'train.txt'), header=None,
                 index=False, sep=",")

    test = pd.DataFrame({'u': test_u, 'i': test_v})
    test.to_csv(os.path.join("..", "..", "FuseRec_SPEX", "data", args.root, "rec", 'test.txt'), header=None,
                index=False, sep=",")

    shutil.copy(os.path.join(args.root, 'type.txt'),
                os.path.join("..", "..", "FuseRec_SPEX", "data", args.root, "rec", 'type.txt'))

    print("to_FuseRec")


def to_LightGCN():
    shutil.copytree(os.path.join("..", "..", "NCF_SPEX", "data", args.root, "rec"),
                    os.path.join("..", "..", "LightGCN_SPEX", "data", args.root, "rec"))
    print("to_LightGCN")


def to_SAMN():
    Path(os.path.join("..", "..", "SAMN_SPEX", "data", args.root, "rec")).mkdir(parents=True, exist_ok=True)
    for item in [(args.root + '.train.rating', 'train.txt'), (args.root + '.test.rating', 'test.txt'),
                 (args.root + '.test.negative', 'negative.txt')]:
        shutil.copy(os.path.join("..", "..", "NCF_SPEX", "data", args.root, "rec", item[0]),
                    os.path.join("..", "..", "SAMN_SPEX", "data", args.root, "rec", item[1]))
    print("to_SAMN")


def to_DiffnetPlus():
    shutil.copytree(os.path.join("..", "..", "NCF_SPEX", "data", args.root, "rec"),
                    os.path.join("..", "..", "Diffnet++_SPEX", "data", args.root, "rec"))
    print("to_DiffnetPlus")


def to_DANSER():
    root = os.path.join("..", "..", "DANSER_SPEX", "data", args.root, "rec")
    Path(root).mkdir(parents=True, exist_ok=True)
    for item in ['train.txt', 'test.txt']:
        shutil.copy(os.path.join("..", "..", "SAMN_SPEX", "data", args.root, "rec", item), os.path.join(root, item))

    # all index + 1
    df_train = pd.read_csv(os.path.join(root, 'train.txt'), sep=" ", names=['uid', 'oid', 'ratting'])
    df_train['uid'] = df_train['uid'] + 1
    df_train['oid'] = df_train['oid'] + 1
    df_train.to_csv(os.path.join(root, 'train.txt'), header=None, index=False, sep=" ")

    df_test = pd.read_csv(os.path.join(root, 'test.txt'), sep=" ", names=['uid', 'oid', 'ratting'])
    df_test['uid'] = df_test['uid'] + 1
    df_test['oid'] = df_test['oid'] + 1
    df_test.to_csv(os.path.join(root, 'test.txt'), header=None, index=False, sep=" ")

    # negative.txt
    # dataset.pkl
    # list.pkl
    # friends.dic
    print("to_DANSER")


if __name__ == '__main__':
    args = parse_args()

    # step 1: output_file: train_rating/test_rating/train_trust/test_trust/type.txt
    change_format()
    select_items()
    select_users()
    unify_index()
    sort()
    split()

    # step 2
    to_NGCF()
    to_NCF()
    to_GraphRec()
    to_FuseRec()
    to_LightGCN()
    to_SAMN()
    to_DiffnetPlus()
    to_DANSER()
