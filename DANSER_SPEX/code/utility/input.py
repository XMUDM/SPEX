import numpy as np
import random
from utility.Logging import Logging
import os
log_dir = os.path.join(os.getcwd(), 'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(os.getcwd(), 'log/test.log')
log = Logging(log_path)
log.record('Following will output the evaluation of the model:')

class DataInput:
    def __init__(self, data, u_read_list, u_friend_list, uf_read_list, i_read_list, i_friend_list, if_read_list, i_link_list, train_batch_size, trunc_len, ui_p, all_i, flag):
        self.train_batch_size = train_batch_size
        self.data = data
        self.u_read_list = u_read_list  
        self.u_friend_list = u_friend_list  
        self.uf_read_list = uf_read_list  
        self.i_read_list = i_read_list
        self.i_friend_list = i_friend_list
        self.if_read_list = if_read_list
        self.i_link_list = i_link_list
        self.epoch_size = len(self.data) // self.train_batch_size
        self.trunc_len = trunc_len
        if self.epoch_size * self.train_batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
        self.ui_p = ui_p
        self.all_i = all_i
        self.flag = flag

        self.s_u_dic = dict()
        self.s_i_dic = dict()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.train_batch_size: min((self.i + 1) * self.train_batch_size, len(self.data))]
        self.i += 1

        iid, uid, label = [], [], []
        u_read, u_friend, uf_read = [], [], []
        u_read_l, u_friend_l, uf_read_l = [], [], []
        i_read, i_friend, if_read = [], [], []
        i_read_l, i_friend_l, if_read_l, i_link = [], [], [], []
        for t in ts:
            uid.append(t[0])
            iid.append(t[1])
            label.append(t[2])

            u_r, u_f, uf_r, u_r_l, u_f_l, uf_r_l = self.s_u(t[0])
            n = 1
            if self.flag: n = 6
            u_read.extend(u_r * n)
            u_friend.extend(u_f * n)
            uf_read.extend(uf_r * n)
            u_read_l.extend(u_r_l * n)
            u_friend_l.extend(u_f_l * n)
            uf_read_l.extend(uf_r_l * n)

            i_r, i_f, if_r, i_r_l, i_f_l, if_r_l, i_l = self.s_i(t[1])
            i_read.extend(i_r)
            i_friend.extend(i_f)
            if_read.extend(if_r)
            i_read_l.extend(i_r_l)
            i_friend_l.extend(i_f_l)
            if_read_l.extend(if_r_l)
            i_link.extend(i_l)

            if self.flag:
                # neg_sample
                opt_items = self.all_i - set(self.ui_p[t[0]])
                for _ in range(5):
                    i_n = random.sample(opt_items, 1)
                    uid.append(t[0])
                    iid.append(i_n[0])
                    label.append(0)

                    i_r, i_f, if_r, i_r_l, i_f_l, if_r_l, i_l = self.s_i(i_n[0])
                    i_read.extend(i_r)
                    i_friend.extend(i_f)
                    if_read.extend(if_r)
                    i_read_l.extend(i_r_l)
                    i_friend_l.extend(i_f_l)
                    if_read_l.extend(if_r_l)
                    i_link.extend(i_l)

        data_len = len(iid)
        # padding
        u_read_maxlength = max(u_read_l)
        u_friend_maxlength = min(self.trunc_len, max(u_friend_l))  # 500
        uf_read_maxlength = min(self.trunc_len, max(max(uf_read_l)))
        u_readinput = np.zeros([data_len, u_read_maxlength], dtype=np.int32)
        for i, ru in enumerate(u_read):
            u_readinput[i, :len(ru)] = ru[:len(ru)]
        u_friendinput = np.zeros([data_len, u_friend_maxlength], dtype=np.int32)
        for i, fi in enumerate(u_friend):
            u_friendinput[i, :min(len(fi), u_friend_maxlength)] = fi[:min(len(fi), u_friend_maxlength)]
        uf_readinput = np.zeros([data_len, u_friend_maxlength, uf_read_maxlength], dtype=np.int32)
        for i in range(len(uf_read)):
            for j, rj in enumerate(uf_read[i][:u_friend_maxlength]):
                uf_readinput[i, j, :min(len(rj), uf_read_maxlength)] = rj[:min(len(rj), uf_read_maxlength)]
        uf_read_linput = np.zeros([data_len, u_friend_maxlength], dtype=np.int32)
        for i, fr in enumerate(uf_read_l):
            uf_read_linput[i, :min(len(fr), u_friend_maxlength)] = fr[:min(len(fr), u_friend_maxlength)]

        i_read_maxlength = max(i_read_l)
        i_friend_maxlength = min(10, max(i_friend_l))  # 500
        if_read_maxlength = min(self.trunc_len, max(max(if_read_l)))
        i_readinput = np.zeros([data_len, i_read_maxlength], dtype=np.int32)
        for i, ru in enumerate(i_read):
            i_readinput[i, :len(ru)] = ru[:len(ru)]
        i_friendinput = np.zeros([data_len, i_friend_maxlength], dtype=np.int32)
        for i, fi in enumerate(i_friend):
            i_friendinput[i, :min(len(fi), i_friend_maxlength)] = fi[:min(len(fi), i_friend_maxlength)]
        if_readinput = np.zeros([data_len, i_friend_maxlength, if_read_maxlength], dtype=np.int32)
        for i in range(len(if_read)):
            for j, rj in enumerate(if_read[i][:i_friend_maxlength]):
                if_readinput[i, j, :min(len(rj), if_read_maxlength)] = rj[:min(len(rj), if_read_maxlength)]
        if_read_linput = np.zeros([data_len, i_friend_maxlength], dtype=np.int32)
        for i, fr in enumerate(if_read_l):
            if_read_linput[i, :min(len(fr), i_friend_maxlength)] = fr[:min(len(fr), i_friend_maxlength)]
        i_linkinput = np.zeros([data_len, i_friend_maxlength, 1], dtype=np.int32)
        for i, li in enumerate(i_link):
            li = np.reshape(np.array(li), [-1, 1])
            i_linkinput[i, :min(len(li), i_friend_maxlength)] = li[:min(len(li), i_friend_maxlength)]
            
        return np.array(uid), np.array(iid), np.array(label), u_readinput, u_friendinput, uf_readinput, np.array(u_read_l), np.array(u_friend_l), uf_read_linput, i_readinput, i_friendinput, if_readinput, i_linkinput, np.array(i_read_l), np.array(i_friend_l), if_read_linput

    def sample(self, data, n_sample):
        loc = []
        select = []
        r = random.randint(0, len(data) - 1)
        for i in range(n_sample):
            while r in loc:
                r = random.randint(0, len(data) - 1)
            loc.append(r)
            select.append(data[r])
        return select

    def s_u(self, u):
        if u in self.s_u_dic:
            return self.s_u_dic[u][0],self.s_u_dic[u][1],self.s_u_dic[u][2],self.s_u_dic[u][3],self.s_u_dic[u][4],self.s_u_dic[u][5]

        u_read, u_friend, uf_read = [], [], []
        u_read_l, u_friend_l, uf_read_l = [], [], []

        u_read_u = self.u_read_list[u]
        u_read.append(u_read_u)
        u_read_l.append(len(u_read_u))

        u_friend_u = self.u_friend_list[u]
        if len(u_friend_u) <= self.trunc_len:
            u_friend.append(u_friend_u)
        else:
            u_friend.append(self.sample(u_friend_u, self.trunc_len))
        u_friend_l.append(min(len(u_friend_u), self.trunc_len))

        uf_read_i = self.uf_read_list[u]  # [[fis],[fis]...]
        uf_read.append(uf_read_i)
        uf_read_l_temp = []
        for f in range(len(uf_read_i)):
            uf_read_l_temp.append(min(len(uf_read_i[f]), self.trunc_len))
        uf_read_l.append(uf_read_l_temp)

        self.s_u_dic[u] = [u_read, u_friend, uf_read, u_read_l, u_friend_l, uf_read_l]

        return u_read, u_friend, uf_read, u_read_l, u_friend_l, uf_read_l

    def s_i(self, i):
        if i in self.s_i_dic:
            return self.s_i_dic[i][0],self.s_i_dic[i][1],self.s_i_dic[i][2],self.s_i_dic[i][3],self.s_i_dic[i][4],self.s_i_dic[i][5],self.s_i_dic[i][6]

        i_read, i_friend, if_read = [], [], []
        i_read_l, i_friend_l, if_read_l, i_link = [], [], [], []

        i_read_u = self.i_read_list[i]
        i_read.append(i_read_u)
        i_read_l.append(len(i_read_u))

        i_friend_i = self.i_friend_list[i]
        if len(i_friend_i) <= self.trunc_len:
            i_friend.append(i_friend_i)
        else:
            i_friend.append(self.sample(i_friend_i, self.trunc_len))
        i_friend_l.append(min(len(i_friend_i), self.trunc_len))

        if_read_u = self.if_read_list[i]
        if_read.append(if_read_u)
        if_read_l_temp = []
        for f in range(len(if_read_u)):
            if_read_l_temp.append(min(len(if_read_u[f]), self.trunc_len))
        if_read_l.append(if_read_l_temp)

        i_link_i = self.i_link_list[i]
        i_link.append(i_link_i)

        self.s_i_dic[i] = [i_read, i_friend, if_read, i_read_l, i_friend_l, if_read_l, i_link]
        return i_read, i_friend, if_read, i_read_l, i_friend_l, if_read_l, i_link
