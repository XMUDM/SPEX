import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility2.layers import GraphAttentionLayer

class FuseRec_SPEX(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, i_class_list):
        super(FuseRec_SPEX, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        i_class_num = len(set(i_class_list))
        self.k = 10
        self.nonhybrid = True
        self.nb_heads = 3

        # trust
        self.linear_one = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.linear_two = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.linear_three = nn.Linear(embedding_dim, 1, bias=False)
        self.linear_transform = nn.Linear(embedding_dim * 2, embedding_dim, bias=True)
        self.reset_parameters()
        self.in_att = [GraphAttentionLayer(embedding_dim, concat=True) for _ in range(self.nb_heads )]
        for i, attention in enumerate(self.in_att):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(embedding_dim, concat=False)
        self.w = nn.Parameter(torch.zeros(size=(self.nb_heads * embedding_dim, embedding_dim)))
        nn.init.xavier_normal_(self.w.data, gain=1.414)
        # expert_s------------------------------------
        self.att_t = nn.Parameter(torch.zeros(size=(2 * embedding_dim, 2)))
        nn.init.xavier_normal_(self.att_t.data, gain=1)
        # -----------------------------------------

        # rec
        self.user_embedding = nn.Embedding(n_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        self.i_class = F.one_hot(torch.tensor(i_class_list).long(), i_class_num).float().cuda()
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=1,batch_first=True)
        self.l1 = nn.Linear(embedding_dim+i_class_num, embedding_dim, bias=True)
        self.l2 = nn.Linear(3 * embedding_dim, embedding_dim, bias=True)
        self.l3 = nn.Linear(embedding_dim+i_class_num, embedding_dim, bias=True)
        self.l4 = nn.Linear(2 * embedding_dim, 1, bias=True)
        self.l5 = nn.Linear(3 * embedding_dim, embedding_dim, bias=True)
        self.l6 = nn.Linear(2 * embedding_dim, 1, bias=True)

        self.lambdas = nn.Parameter(torch.FloatTensor([0.5, 0.5, 0.5, 0.5]))
        self.alpha = nn.Parameter(torch.FloatTensor([0.5]))  # [0,1]

        self.rec_loss = nn.BCELoss()
        self.tru_loss = nn.CrossEntropyLoss()
        self.task_weights = nn.Parameter(torch.FloatTensor([0.5, 0.5]))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reg_loss(self):
        return torch.norm(input=self.user_embedding.weight, p=2) + torch.norm(input=self.item_embedding.weight, p=2)

    def forward(self, users, items, labels, u_items, u_items_mask, u_frids, u_frids_mask, u_frids_items, F_i, slice_indices, data, flag=0):

        if (flag == 0) or (flag == 1):
            u = self.user_embedding(users)
            i = self.item_embedding(items)
            v = self.user_embedding(u_frids)

            # User-temporal module
            ju_all = self.l1(torch.cat([self.item_embedding.weight, self.i_class], dim=1))  # ju
            ju = ju_all[u_items]
            output, _ = self.lstm(ju)  # [B,D]
            hu = torch.stack([output[i_ind][j_ind-1] for i_ind, j_ind in enumerate(u_items_mask)], 0)
            hui = self.l2(torch.cat([hu, i, hu * i],dim=1))  # [B,D]

            # User-social module
            jv_all = self.l3(torch.cat([self.item_embedding.weight, self.i_class], dim=1))  # jv
            jv = jv_all[u_frids_items]  # [B, u_f_l, u_f_i_l, D]
            m = u_frids_mask.unsqueeze(1)
            pv = torch.sum(jv, dim=2) / torch.stack([m] * self.k, 1).float()  # [B, u_f_l, D]
            u_rep = u.repeat(1, self.k).reshape(-1, self.k, self.embedding_dim)
            at = F.leaky_relu(self.l4(torch.cat([u_rep, v], dim=2))).squeeze()
            auv = torch.stack([torch.cat([F.softmax(at_u[:at_l], dim=0), torch.zeros(10 - at_l).cuda()], 0) for at_u, at_l in zip(at, u_frids_mask)], 0)  # [B,u_f_l]
            su = torch.sum(pv * auv.unsqueeze(2), dim=1)  # [B,D]
            sui = self.l5(torch.cat([su, i, su * i], dim=1))  # [B,D]

            # Item-similarity module
            Fi_1, Fi_2 = self.item_embedding(F_i[:,0,:]), self.item_embedding(F_i[:,1,:])  # [B, K]
            i_rep = i.repeat(1, self.k).reshape(-1, self.k, self.embedding_dim)  # [B, K, D]
            bij_1 = F.softmax(F.leaky_relu(self.l6(torch.cat([i_rep, Fi_1], dim=2))).squeeze(), dim=0).unsqueeze(2)  # [B, K]
            bij_2 = F.softmax(F.leaky_relu(self.l6(torch.cat([i_rep, Fi_2], dim=2))).squeeze(), dim=0).unsqueeze(2)
            yi = self.alpha * torch.sum(bij_1 * Fi_1, 1) + (1 - self.alpha) * torch.sum(Fi_2 * bij_2 ,1) # [B,D]

            # Compute score
            S1 = torch.sum(hu * yi, 1)  # inner product # [B,]
            S2 = torch.sum(hui * yi, 1)
            S3 = torch.sum(su * yi, 1)
            S4 = torch.sum(sui * yi, 1)
            rui = F.sigmoid(self.lambdas[0] * S1 + self.lambdas[1] * S2 + self.lambdas[2] * S3 + self.lambdas[3] * S4)

            if flag == 1: return rui
            loss1 = self.rec_loss(rui, labels.float())
        if (flag == 0) or (flag == 2):
            if flag == 0:
                inputs, mask, targets = data.get_slice(slice_indices)
            else:
                inputs, mask, targets, negs = data.get_slice(slice_indices)
                negs = torch.Tensor(negs).long().cuda()

            inputs = torch.Tensor(inputs).long().cuda()
            mask = torch.Tensor(mask).long().cuda()
            targets = torch.Tensor(targets).long().cuda()

            # multi head
            seq_l = torch.sum(mask, 1)
            mul_seq = torch.cat([att(self.user_embedding.weight, inputs, seq_l) for att in self.in_att], dim=2).cuda()
            mul_seq_c = torch.cat([mul_seq[i] for i in range(mul_seq.size()[0])], dim=0)
            mul_one = torch.mm(mul_seq_c, self.w)
            mul_one = F.elu(mul_one)
            seq_hidden = self.out_att(self.user_embedding.weight,mul_one.view(mul_seq.size()[0], mul_seq.size()[1],self.embedding_dim),seq_l).cuda()

            trust_scores = self.compute_trust_scores(seq_hidden, inputs, mask)
            if flag == 2:
                return trust_scores, negs
            loss2 = self.tru_loss(trust_scores, targets)
        return loss1, loss2

    def compute_trust_scores(self, hidden, inputs, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        p_a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            p_a = self.linear_transform(torch.cat([p_a, ht], 1))
        b = self.user_embedding.weight[:-1]

        # ------------------------------
        p_i_all = self.user_embedding.weight[inputs]
        p_i = p_i_all * mask.unsqueeze(2)
        p_maxpool = torch.max(p_i, dim=1)[0]
        
        att = torch.softmax(torch.matmul(torch.cat([p_a, p_maxpool], 1), self.att_t), 1)  # [B,2H] x [2H,2] = [B,2]
        a = p_a * att[:, 0].unsqueeze(1) + p_maxpool * att[:, 1].unsqueeze(1)
        # ------------------------------

        scores = torch.matmul(a, b.transpose(1, 0))
        return scores
