import torch.utils.data
import torch.nn.functional as F
from torch import nn

class DANSER(nn.Module):

    def __init__(self, user_count, item_count, args, training=True):
        super(DANSER, self).__init__()
        # --------------embedding layer-------------------
        self.hidden_units_u = args.embedding_size # user embedding size
        self.hidden_units_i = args.embedding_size  # item embedding size

        self.user_emb_w = nn.Embedding(user_count+1, self.hidden_units_u, padding_idx=0)
        self.item_emb_w = nn.Embedding(item_count+1, self.hidden_units_i, padding_idx=0)
        self.item_b = nn.Parameter(torch.FloatTensor([0.0] * (item_count+1)))

        self.training = training
        self.keep_prob = args.keep_prob
        self.lambda1 = 0.001
        self.lambda2 = 0.001

        self.trans_uid = nn.Linear(self.hidden_units_u, self.hidden_units_u, bias=False)
        self.trans_iid = nn.Linear(self.hidden_units_i, self.hidden_units_i, bias=False)
        self.gat_uid = nn.Linear(self.hidden_units_u * 2, 1)
        self.gat_iid = nn.Linear(self.hidden_units_i * 2, 1)
        self.trans_uint = nn.Linear(self.hidden_units_i, self.hidden_units_i, bias=False)
        self.trans_iinf = nn.Linear(self.hidden_units_u, self.hidden_units_u, bias=False)
        self.gat_uint = nn.Linear(self.hidden_units_i * 2, 1)
        self.gat_iinf = nn.Linear(self.hidden_units_u * 2, 1)

        self.norm_ui = nn.Sequential(
            nn.Linear(self.hidden_units_u, 16, bias=True),
            nn.Tanh(),
            nn.Dropout(self.keep_prob),
            nn.Linear(16, 8, bias=True),
            nn.Tanh(),
            nn.Dropout(self.keep_prob),
            nn.Linear(8, 4, bias=True),
            nn.Tanh(),
        )
        self.norm_uf = nn.Sequential(
            nn.Linear(self.hidden_units_u, 16, bias=True),
            nn.Tanh(),
            nn.Dropout(self.keep_prob),
            nn.Linear(16, 8, bias=True),
            nn.Tanh(),
            nn.Dropout(self.keep_prob),
            nn.Linear(8, 4, bias=True),
            nn.Tanh(),
        )
        self.norm_fi = nn.Sequential(
            nn.Linear(self.hidden_units_i, 16, bias=True),
            nn.Tanh(),
            nn.Dropout(self.keep_prob),
            nn.Linear(16, 8, bias=True),
            nn.Tanh(),
            nn.Dropout(self.keep_prob),
            nn.Linear(8, 4, bias=True),
            nn.Tanh(),
        )
        self.norm_ff = nn.Sequential(
            nn.Linear(self.hidden_units_i, 16, bias=True),
            nn.Tanh(),
            nn.Dropout(self.keep_prob),
            nn.Linear(16, 8, bias=True),
            nn.Tanh(),
            nn.Dropout(self.keep_prob),
            nn.Linear(8, 4, bias=True),
            nn.Tanh(),
        )
        self.norm_merge = nn.Linear(4, 1, bias=True)

        self.policy_1 = nn.Linear(self.hidden_units_u * 3, 4)
        self.policy_2 = nn.Linear(self.hidden_units_u * 3, 4)
        self.policy_3 = nn.Linear(self.hidden_units_u * 3, 4)
        self.policy_4 = nn.Linear(self.hidden_units_u * 3, 4)

        self.norm_ui_b1 = nn.BatchNorm1d(num_features=self.hidden_units_u)
        self.norm_uf_b1 = nn.BatchNorm1d(num_features=self.hidden_units_u)
        self.norm_fi_b1 = nn.BatchNorm1d(num_features=self.hidden_units_u)
        self.norm_ff_b1 = nn.BatchNorm1d(num_features=self.hidden_units_u)

        self.dropout = nn.Dropout(self.keep_prob)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, u, i, l, u_read, u_friend, uf_read,u_read_l,u_friend_l,uf_read_l, i_read, i_friend, if_read, i_link, i_read_l, i_friend_l, if_read_l, flag):
        user = u
        item = i
        label = l
        uid_emb = self.user_emb_w(user)
        iid_emb = self.item_emb_w(item)
        i_b = self.item_b[item]

        # embedding for user's clicked items
        ur_emb = self.item_emb_w(u_read) # [B, R, H]
        key_masks = torch.tensor([[True]*i + [False]*(ur_emb.shape[1]-i) for i in u_read_l]).cuda()
        key_masks = key_masks.unsqueeze(2)  # [B, R, 1]
        key_masks = key_masks.repeat(1, 1, ur_emb.shape[2])  # [B, R, H]
        key_masks = key_masks.reshape(-1, ur_emb.shape[1], ur_emb.shape[2]).cuda()  # [B, R, H]
        paddings = torch.zeros_like(ur_emb).cuda()  # [B, R, H]
        ur_emb = torch.where(key_masks, ur_emb, paddings).cuda()  #  [ 64 276  10] [ 64 276  10] [ 64 276  10]

        # embedding for item's clicking users
        ir_emb = self.user_emb_w(i_read)  # [B, R, H]
        key_masks = torch.tensor([[True]*i + [False]*(ir_emb.shape[1]-i) for i in i_read_l]).cuda() # [B, R]
        key_masks = key_masks.unsqueeze(2)  # [B, R, 1]
        key_masks = key_masks.repeat(1, 1, ir_emb.shape[2])  # [B, R, H]
        key_masks = key_masks.reshape(-1, ir_emb.shape[1], ir_emb.shape[2])  # [B, R, H]
        paddings = torch.zeros_like(ir_emb).cuda()  # [B, R, H]
        ir_emb = torch.where(key_masks, ir_emb, paddings) # [B, R, H]

        # embedding for user's friends
        fuid_emb = self.user_emb_w(u_friend)
        key_masks = torch.tensor([[True]*i + [False]*(fuid_emb.shape[1]-i) for i in u_friend_l]).cuda() # [B, F]
        key_masks = key_masks.unsqueeze(2)  # [B, F, 1]
        key_masks = key_masks.repeat(1, 1, fuid_emb.shape[2])  # [B, F, H]
        paddings = torch.zeros_like(fuid_emb).cuda()  # [B, F, H]
        fuid_emb = torch.where(key_masks, fuid_emb, paddings)  # [B, F, H]

        # embedding for item's related items
        fiid_emb = self.item_emb_w(i_friend)
        key_masks = torch.tensor([[True]*i + [False]*(fiid_emb.shape[1]-i) for i in i_friend_l]).cuda()  # [B, F]
        key_masks = key_masks.unsqueeze(2)  # [B, F, 1]
        key_masks = key_masks.repeat(1, 1, fiid_emb.shape[2])  # [B, F, H]
        paddings = torch.zeros_like(fiid_emb).cuda()  # [B, F, H]
        fiid_emb = torch.where(key_masks.byte(), fiid_emb, paddings)  # [B, F, H]

        # embedding for user's friends' clicked items
        ufr_emb = self.item_emb_w(uf_read) # [384, 10, 1020, 10]
        k = torch.zeros(size=(uf_read_l.shape[0], uf_read_l.shape[1], ufr_emb.shape[2]))
        for i in range(uf_read_l.shape[0]):
            for j in range(uf_read_l.shape[1]):
                t = uf_read_l[i][j]
                k[i][j][:t] = 1
        key_masks = k.bool()
        key_masks = key_masks.unsqueeze(3)  # [B, F, R, 1]
        key_masks = key_masks.repeat(1, 1, 1, ufr_emb.shape[3]).cuda()  # [B, F, R, H]
        paddings = torch.zeros_like(ufr_emb).cuda()  # [B, F, R, H]
        ufr_emb = torch.where(key_masks, ufr_emb, paddings)  # [B, F, R, H]

        # embedding for item's related items' clicking users
        ifr_emb = self.user_emb_w(if_read)  # [B, F, R, H]  (384, 0, 991, 10)
        k = torch.zeros(size=(if_read_l.shape[0], if_read_l.shape[1], ifr_emb.shape[2]))
        for i in range(if_read_l.shape[0]):
            for j in range(if_read_l.shape[1]):
                t = if_read_l[i][j]
                k[i][j][:t] = 1
        key_masks = k.bool()
        key_masks = key_masks.unsqueeze(3)  # [B, F, R, 1]
        key_masks = key_masks.repeat(1, 1, 1, ifr_emb.shape[3]).cuda()  # [B, F, R, H]
        paddings = torch.zeros_like(ifr_emb).cuda()  # [B, F, R, H]
        ifr_emb = torch.where(key_masks, ifr_emb, paddings)  # [B, F, R, H]

        # --------------social influence-------------------

        uid_emb_exp1 = uid_emb.repeat(1, fuid_emb.shape[1] + 1)
        uid_emb_exp1 = uid_emb_exp1.reshape(-1, fuid_emb.shape[1] + 1, self.hidden_units_u)  # [B, F, H]
        iid_emb_exp1 = iid_emb.repeat(1, fiid_emb.shape[1] + 1)
        iid_emb_exp1 = iid_emb_exp1.reshape(-1, fiid_emb.shape[1] + 1, self.hidden_units_i)  # [B, F, H]
        uid_emb_ = uid_emb.unsqueeze(1)
        iid_emb_ = iid_emb.unsqueeze(1)

        # GAT1: graph convolution on user's embedding for user static preference
        uid_in = self.trans_uid(uid_emb_exp1)
        fuid_in = self.trans_uid(torch.cat((uid_emb_, fuid_emb), 1))
        din_gat_uid = torch.cat((uid_in, fuid_in), -1)
        d1_gat_uid = self.gat_uid(din_gat_uid)
        d1_gat_uid = F.leaky_relu(d1_gat_uid, inplace=True)
        d1_gat_uid = self.dropout(d1_gat_uid)
        d1_gat_uid = d1_gat_uid.reshape(-1, ufr_emb.shape[1] + 1, 1)  # [B, F, 1]
        weights_uid = F.softmax(d1_gat_uid, 1)  # [B, F, 1]
        weights_uid = weights_uid.repeat(1, 1, self.hidden_units_u)  # [B, F, H]
        uid_gat = torch.sum(torch.mul(weights_uid, fuid_in), 1)
        uid_gat = uid_gat.reshape(-1, self.hidden_units_u)

        # GAT2: graph convolution on item's embedding for item static attribute
        iid_in = self.trans_iid(iid_emb_exp1)
        fiid_in = self.trans_iid(torch.cat((iid_emb_, fiid_emb), 1))
        din_gat_iid = torch.cat((iid_in, fiid_in), -1)
        d1_gat_iid = self.gat_iid(din_gat_iid)
        d1_gat_iid = F.leaky_relu(d1_gat_iid, inplace=True)
        d1_gat_iid = self.dropout(d1_gat_iid)
        d1_gat_iid = d1_gat_iid.reshape(-1, ifr_emb.shape[1] + 1, 1)  # [B, F, 1]
        weights_iid = F.softmax(d1_gat_iid, 1)  # [B, F, 1]
        weights_iid = weights_iid.repeat(1, 1, self.hidden_units_i)  # [B, F, H]
        iid_gat = torch.sum(torch.mul(weights_iid, fiid_in), 1) # reducesum
        iid_gat = iid_gat.reshape(-1, self.hidden_units_i)

        uid_emb_exp2 = uid_emb.repeat(1, ir_emb.shape[1])
        uid_emb_exp2 = uid_emb_exp2.reshape(-1, ir_emb.shape[1], self.hidden_units_u)  # [B, R, H]
        iid_emb_exp2 = iid_emb.repeat(1, ur_emb.shape[1])
        iid_emb_exp2 = iid_emb_exp2.reshape(-1, ur_emb.shape[1], self.hidden_units_i)  # [B, R, H]
        uid_emb_exp3 = uid_emb.unsqueeze(1)
        uid_emb_exp3 = uid_emb_exp3.unsqueeze(2) # [B, 1, 1, H]  ([384, 1, 1, 10])
        uid_emb_exp3 = uid_emb_exp3.repeat(1, ifr_emb.shape[1], ifr_emb.shape[2], 1)  # [B, F, R, H]  ([384, 0, 991, 10])
        iid_emb_exp3 = iid_emb.unsqueeze(1)
        iid_emb_exp3 = iid_emb_exp3.unsqueeze(2) # [B, 1, 1, H]
        iid_emb_exp3 = iid_emb_exp3.repeat(1, ufr_emb.shape[1], ufr_emb.shape[2], 1)  # [B, F, R, H]

        # GAT3: graph convolution on user's clicked items for user dynamic preference
        uint_in = torch.mul(ur_emb, iid_emb_exp2)  # [B, R, H]
        uint_in = torch.max(uint_in, 1)[0]  # [B, H]   reducemax
        uint_in = self.trans_uint(uint_in)  # [B, H]
        uint_in_ = uint_in.unsqueeze(1)  # [B, 1, H]
        uint_in = uint_in.repeat(1, ufr_emb.shape[1] + 1)
        uint_in = uint_in.reshape(-1, ufr_emb.shape[1] + 1, self.hidden_units_i)  # [B, F, H]
        fint_in = torch.mul(ufr_emb, iid_emb_exp3)  # [B, F, R, H]
        fint_in = torch.max(fint_in, 2)[0]  # [B, F, H]
        fint_in = self.trans_uint(fint_in)
        fint_in = torch.cat((uint_in_, fint_in), 1)  # [B, F, H]
        din_gat_uint = torch.cat((uint_in, fint_in), -1)
        d1_gat_uint = self.gat_uint(din_gat_uint)
        d1_gat_uint = F.leaky_relu(d1_gat_uint, inplace=True)
        d1_gat_uint = self.dropout(d1_gat_uint)
        d1_gat_uint = d1_gat_uint.reshape(-1, ufr_emb.shape[1] + 1, 1)  # [B, F, 1]
        weights_uint = F.softmax(d1_gat_uint, 1)  # [B, F, 1]
        weights_uint = weights_uint.repeat(1, 1, self.hidden_units_i)  # [B, F, H]
        uint_gat = torch.sum(torch.mul(weights_uint, fint_in), 1)
        uint_gat = uint_gat.reshape(-1, self.hidden_units_i)

        # GAT4: graph convolution on item's clicking users for item dynamic attribute
        iinf_in = torch.mul(ir_emb, uid_emb_exp2)  # [B, R, H]
        iinf_in = torch.max(iinf_in, 1)[0]  # [B, H]
        iinf_in = self.trans_iinf(iinf_in)  # [B, H]
        iinf_in_ = iinf_in.unsqueeze(1)  # [B, 1, H]
        iinf_in = iinf_in.repeat(1, ifr_emb.shape[1] + 1)
        iinf_in = iinf_in.reshape(-1, ifr_emb.shape[1] + 1, self.hidden_units_u)  # [B, F, H]
        finf_in = torch.mul(ifr_emb, uid_emb_exp3)  # [B, F, R, H]  (64, 0, 1629, 10)
        if finf_in.shape[1] == 0:
            finf_in = torch.zeros(finf_in.shape[0], finf_in.shape[1], finf_in.shape[3]).cuda()
        else:
            finf_in = torch.max(finf_in, 2)[0]  # [B, F, H]  RuntimeError: cannot perform reduction function max on tensor with no elements because the operation does not have an identity
        finf_in = self.trans_iinf(finf_in)
        finf_in = torch.cat((iinf_in_, finf_in), 1)  # [B, F, H]
        din_gat_iinf = torch.cat((iinf_in, finf_in), -1)
        d1_gat_iinf = self.gat_iinf(din_gat_iinf)
        d1_gat_iinf = F.leaky_relu(d1_gat_iinf, inplace=True)
        d1_gat_iinf = self.dropout(d1_gat_iinf)
        d1_gat_iinf = d1_gat_iinf.reshape(-1, ifr_emb.shape[1] + 1, 1)  # [B, F, 1]
        weights_iinf = F.softmax(d1_gat_iinf, 1)  # [B, F, 1]
        weights_iinf = weights_iinf.repeat(1, 1, self.hidden_units_u)  # [B, F, H]
        iinf_gat = torch.sum(torch.mul(weights_iinf, finf_in), 1)
        iinf_gat = iinf_gat.reshape(-1, self.hidden_units_u)

        # --------------DNN-based pairwise neural interaction layer---------------
        din_ui = torch.mul(uid_gat, iid_gat)
        din_ui = self.norm_ui_b1(din_ui)
        d3_ui = self.norm_ui(din_ui)
        d3_ui_ = d3_ui.reshape(-1, d3_ui.shape[-1], 1)

        din_uf = torch.mul(uid_gat, iinf_gat)
        din_uf = self.norm_uf_b1(din_uf)
        d3_uf = self.norm_uf(din_uf)
        d3_uf_ = d3_uf.reshape(-1, d3_uf.shape[-1], 1)

        din_fi = torch.mul(uint_gat, iid_gat)
        self.norm_fi_b1(din_fi)
        d3_fi= self.norm_fi(din_fi)
        d3_fi_ = d3_fi.reshape(-1, d3_fi.shape[-1], 1)

        din_ff = torch.mul(uint_gat, iinf_gat)
        din_ff = self.norm_ff_b1(din_ff)
        d3_ff = self.norm_ff(din_ff)
        d3_ff_ = d3_ff.reshape(-1, d3_ff.shape[-1], 1)

        d3 = torch.cat((d3_ui_, d3_uf_, d3_fi_, d3_ff_), 2)

        # --------------policy-based fusion layer---------------
        din_policy = torch.cat((uid_emb, iid_emb, torch.mul(uid_emb, iid_emb)), -1)  # B x 3H
        policy1 = F.softmax(self.policy_1(din_policy), dim=0)  # B x 4
        policy2 = F.softmax(self.policy_2(din_policy), dim=0)
        policy3 = F.softmax(self.policy_3(din_policy), dim=0)
        policy4 = F.softmax(self.policy_4(din_policy), dim=0)
        policy = (policy1 + policy2 + policy3 + policy4) / 4  
        policy_exp = policy.repeat(1, d3_ui.shape[-1])   # B x 16
        policy_exp = policy_exp.reshape(-1, d3_ui.shape[-1], 4)    # B x 4 x 4
        if self.training == True:
            dist = torch.multinomial(input=policy, num_samples=1)  
            t = torch.zeros_like(policy)
            for i in range(len(dist)):
                j = dist[i][0]

                t[i][j] = 1
            t = t.reshape(-1, 4)  # [B, 4]
            t_exp = t.repeat(1, d3_ui.shape[-1])
            t_exp = t_exp.reshape(-1, d3_ui.shape[-1], 4)
            dmerge = torch.sum(torch.mul(t_exp, d3), 2)
        else:
            dmerge = torch.sum(torch.mul(policy_exp, d3), 2)
        dmerge = dmerge.reshape(-1, 4)
        dmerge = self.norm_merge(dmerge)
        dmerge = F.sigmoid(dmerge) 
        dmerge = dmerge.reshape(-1)


        # --------------output layer---------------
        self.score = i_b + dmerge
        if flag == 0:
            return self.score
        else:
            loss = self.criterion(self.score, label.float())
            return loss