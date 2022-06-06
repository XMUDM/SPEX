import shutil
from collections import defaultdict
import os
import pickle
import random
import pandas as pd
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='epinion2', help='epinion2/weibo/twitter')
    parser.add_argument('--m', type=int, default=50)
    parser.add_argument('--path_len', type=int, default=6)
    parser.add_argument('--user_num', type=int, default=100000)
    parser.add_argument('--item_num', type=int, default=1000000)
    parser.add_argument('--max_count', type=int, default=10000)
    parser.add_argument('--max_time', type=int, default=13,
                        help="epinion2:13,weibo/twitter:22220000000000,for padding")
    parser.add_argument('--fix_len', type=int, default=769,
                        help="epinion2:769,weibo:178,twitter:115,padding length")

    return parser.parse_args()


def process_selftrust():
    '''
    delete lines like 'user_A trust user_A'
    '''
    users_t = defaultdict(list)
    with open(os.path.join(args.root, 'trust.txt'), 'r') as f:
        for line in f:
            arr = line.strip().split()
            user, friend, time = arr[0], arr[1], arr[2]
            if user != friend:
                users_t[user].append([friend, time])

    with open(os.path.join(args.root, 'trust2.txt'), 'w') as f:
        for u in users_t:
            for ft in users_t[u]:
                f.write(u + ' ' + ft[0] + ' ' + ft[1] + '\n')

    print("process_selftrust")


def built_user():
    users_t = defaultdict(list)  # users_t = {uid:[(fid,time),(fid,time),...],uid:[(fid,time),(fid,time),...],...}
    users_f = defaultdict(list)  # users_f = {uid:[fid,fid,...],uid:[fid,fid,...],...}
    with open(os.path.join(args.root, 'trust2.txt'), 'r') as f:  # trust2.txt 无自己信任自己
        for line in f:
            arr = line.strip().split()
            user, friend, time = arr[0], arr[1], arr[2]
            users_t[user].append((friend, time))
            users_f[user].append(friend)

    with open(os.path.join(args.root, 'users_t.dic'), 'wb') as f:
        pickle.dump(users_t, f)
    with open(os.path.join(args.root, 'users_f.dic'), 'wb') as f:
        pickle.dump(users_f, f)

    print("build user !")


def build_path():
    '''
    Random path sampling
    '''

    with open(os.path.join(args.root, 'users_t.dic'), 'rb') as f:
        users_t = pickle.load(f, encoding='bytes')

    path_trust = defaultdict(list)
    for user in users_t:
        for i in range(args.m):
            cur_path = [(user, 0)]
            while len(cur_path) < args.path_len:
                u_id = cur_path[-1][0]
                u_time = cur_path[-1][1]
                if u_id not in users_t: break
                f = random.choice(users_t[u_id])
                count = 0
                while (int(f[1]) <= int(u_time)):  # time order
                    f = random.choice(users_t[u_id])
                    count += 1
                    if count >= args.max_count:
                        break
                if count == args.max_count:
                    while len(cur_path) < args.path_len:
                        cur_path.append((args.user_num, args.max_time))
                else:
                    cur_path.append(f)
            while len(cur_path) < args.path_len:
                cur_path.append((args.user_num, args.max_time))

            temp = [int(t[0]) for t in cur_path]
            path_trust[user].append(temp)

    '''
    Remove duplicate paths
    '''

    p0, p1, p2, p3, p4, p5 = [], [], [], [], [], []
    for u in path_trust:
        for path in path_trust[u]:
            p0.append(path[0])
            p1.append(path[1])
            p2.append(path[2])
            p3.append(path[3])
            p4.append(path[4])
            p5.append(path[5])

    df = pd.DataFrame({"p0": p0, "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5})

    data2 = df.drop_duplicates(keep='first')
    print("trust path num:", len(data2))
    data = data2.values.tolist()

    new_p = defaultdict(list)
    for p in data:
        new_p[p[0]].append(p)

    with open(os.path.join(args.root, 'path_trust.dic'), "wb") as f:
        pickle.dump(new_p, f)

    print("build path !")


def data_enhancement():
    '''
    1. Delete the supplementary elements.
    2. Separate the training set and test set.
    3. Only do data enhancement for training set.
    '''

    with open(os.path.join(args.root, 'path_trust.dic'), 'rb') as f:
        path_trust = pickle.load(f, encoding='bytes')

    # step1
    count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for u in path_trust:
        for i, path in enumerate(path_trust[u]):
            if int(path[1]) == args.user_num:
                count[1] += 1
                del path_trust[u][i][1:]
            elif int(path[2]) == args.user_num:
                count[2] += 1
                del path_trust[u][i][2:]
            elif int(path[3]) == args.user_num:
                count[3] += 1
                del path_trust[u][i][3:]
            elif int(path[4]) == args.user_num:
                count[4] += 1
                del path_trust[u][i][4:]
            elif int(path[5]) == args.user_num:
                count[5] += 1
                del path_trust[u][i][5:]

    # step2
    train = []
    test_data, test_target = [], []
    a = defaultdict(set)
    for u in path_trust:
        l = len(path_trust[u])
        for i, path in enumerate(path_trust[u]):
            if i < int(l * 0.9):  # 0.9:100%  0.72:80%  0.45:50%
                train.append(path)
                for i in range(len(path) - 1):
                    a[path[i]].add(path[i + 1])
            else:
                test_data.append(path[:-1])
                test_target.append(path[-1])

    # step3
    def process_seqs(iseqs):
        out_seqs = []
        labs = []
        for id, seq in zip(range(len(iseqs)), iseqs):
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                out_seqs += [seq[:-i]]
        return out_seqs, labs

    train_data, train_target = process_seqs(train)

    with open(os.path.join(args.root, 'train.txt'), 'wb') as f:
        pickle.dump((train_data, train_target), f)

    with open(os.path.join(args.root, 'test.txt'), 'wb') as f:
        pickle.dump((test_data, test_target), f)

    print("data_enhancement")


def neg_sample():
    test = pickle.load(open(os.path.join(args.root, 'test.txt'), 'rb'))
    u_f2 = pickle.load(open(os.path.join(args.root, 'users_f.dic'), 'rb'))
    all_user = set(range(args.user_num))

    neg_list2 = []
    for path, t in zip(test[0], test[1]):
        neg = all_user
        neg = neg.difference(set(u_f2[path[-1]]))
        n_p = random.sample(neg, 499)
        n_p.append(t)
        neg_list2.append(n_p)
    with open(os.path.join(args.root, 'test2.txt'), 'wb') as f:
        pickle.dump((test[0], test[1], neg_list2), f)

    print("neg_sample")


def data_refine():
    ori_data = pickle.load(open(os.path.join(args.root, 'train.txt'), 'rb'))
    ss = set()
    a, b, refine_data = [], [], []
    for i, j in zip(ori_data[0], ori_data[1]):
        t = (tuple(i), j)
        if t not in ss:
            ss.add(t)
            a.append(i)
            b.append(j)
    refine_data.append(a)
    refine_data.append(b)
    pickle.dump(refine_data, open(os.path.join(args.root, 'train.txt'), 'wb'))

    print("data_refine")


def copy_file():

    for model_name in ['NCF_SPEX', 'NGCF_SPEX', 'Diffnet++_SPEX', 'FuseRec_SPEX', 'GraphRec_SPEX', 'LightGCN_SPEX',
                       'SAMN_SPEX', 'DANSER_SPEX']:
        root = os.path.join("..", "..", model_name, "data", args.root, "trust")
        Path(root).mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(args.root, "train.txt"), os.path.join(root, "train.txt"))
        shutil.copy(os.path.join(args.root, "test.txt"), os.path.join(root, "test.txt"))
        shutil.copy(os.path.join(args.root, "test2.txt"), os.path.join(root, "test2.txt"))
        shutil.copy(os.path.join(args.root, "users_f.dic"), os.path.join(root, "users_f.dic"))

    print("copy_file")


def copy_file2():
    # friends.dic、social.share
    max_user = 0
    u_f = pickle.load(open(os.path.join(args.root, 'users_f.dic'), 'rb'))
    for u in u_f:
        max_user = max(max_user, int(max(u_f[u])))
    max_user += 1

    for u in u_f:
        if len(u_f[u]) < args.fix_len:
            u_f[u].extend([str(max_user)] * (args.fix_len - len(u_f[u])))
        else:
            u_f[u] = u_f[u][:args.fix_len]

    with open(os.path.join(args.root, 'friends.dic'), "wb") as f:
        pickle.dump(u_f, f)

    with open(os.path.join(args.root, "social.share"), "w") as f:
        for u in u_f:
            for i in u_f[u]:
                f.write(u + " " + i + "\n")

    # SAMN
    shutil.copy(os.path.join(args.root, "friends.dic"),
                os.path.join("..", "..", "SAMN_SPEX", "data", args.root, "rec", "friends.dic"))

    # Diffnet++
    shutil.copy(os.path.join(args.root, "social.share"),
                os.path.join("..", "..", "Diffnet++_SPEX", "data", args.root, "rec", "social.share"))

    print("copy_file2")


if __name__ == '__main__':
    args = parse_args()

    # step 1
    # Path(args.root).mkdir(parents=True, exist_ok=True)
    # shutil.copy(os.path.join("..", "rec", args.root, "train_trust.txt"), os.path.join(args.root, "trust.txt"))
    # process_selftrust()
    # built_user()
    # build_path()
    # data_enhancement()
    # neg_sample()
    # data_refine()

    # step 2
    # copy_file()
    copy_file2()
