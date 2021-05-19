import math
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
from utility2.layers import GraphAttentionLayer


class SAMN_TRUST(nn.Module):

    def __init__(self, user_num, item_num, args):
        super(SAMN_TRUST, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = args.embedding_size

        # trust
        self.batch_size = args.batch_size
        self.nonhybrid = args.nonhybrid
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_size * 2, self.embedding_size, bias=True)
        self.reset_parameters()

        self.in_att = [GraphAttentionLayer(self.embedding_size, concat=True) for _ in range(args.nb_heads)]
        for i, attention in enumerate(self.in_att):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(self.embedding_size, concat=False)
        self.w = nn.Parameter(torch.zeros(size=(args.nb_heads * self.embedding_size, self.embedding_size)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.criterion2 = nn.CrossEntropyLoss()

        # rec
        self.attention_size = args.attention_size
        self.memory_size = args.memory_size
        self.uidW = nn.Parameter(torch.FloatTensor(user_num + 1, self.embedding_size).uniform_(-0.1, 0.1))
        self.iidW = nn.Parameter(torch.FloatTensor(item_num, self.embedding_size).uniform_(-0.1, 0.1))
        self.i_bias = nn.Parameter(torch.FloatTensor([0.0] * self.item_num))
        self.Key = nn.Parameter(torch.FloatTensor(self.embedding_size, self.memory_size).uniform_(-0.1, 0.1))
        self.Mem = nn.Parameter(torch.FloatTensor(self.memory_size, self.embedding_size).uniform_(1.0, 1.0))
        self.WA = nn.Parameter(torch.FloatTensor(self.embedding_size, self.attention_size).uniform_(-0.1, 0.1))
        self.BA = nn.Parameter(torch.zeros(self.attention_size))
        self.U_omega = nn.Parameter(torch.FloatTensor(self.attention_size, 1).uniform_(-0.1, 0.1))
        self.dropout = nn.Dropout(p=args.dp)
        self.criterion1 = nn.BCEWithLogitsLoss()

        # multi_task
        self.task_weights = nn.Parameter(torch.FloatTensor([0.5, 0.5]))
        # expert_s------------------------------------
        self.att_exp1 = nn.Parameter(torch.zeros(size=(2 * self.embedding_size, 2)))
        nn.init.xavier_uniform_(self.att_exp1.data, gain=1)
        self.att_t = nn.Parameter(torch.zeros(size=(2 * self.embedding_size, 2)))
        nn.init.xavier_normal_(self.att_t.data, gain=1)
        # -----------------------------------------

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_u, input_i, label, input_uf, i, data, flag):
        if (flag == 1) or (flag == 0):
            self.uid = self.uidW[input_u]
            self.iid = self.iidW[input_i]
            self.uid = self.uid.reshape([-1, self.embedding_size])
            self.iid = self.iid.reshape([-1, self.embedding_size])
            self.i_b = self.i_bias[input_i]
            # memory_attention
            self.frien_embedding = self.uidW[input_uf]
            self.frien_num = (input_uf.eq(self.user_num) == False).float()
            self.frien_embedding = torch.einsum('ab,abc->abc', [self.frien_num, self.frien_embedding])
            self.uid_n = torch.nn.functional.normalize(self.uid, p=2, dim=1)
            self.frien_embedding_n = torch.nn.functional.normalize(self.frien_embedding, p=2, dim=2)
            self.cross_friend = torch.einsum('ac,abc->abc', self.uid_n, self.frien_embedding_n)
            self.att_key = torch.einsum('abc,ck->abk', [self.cross_friend, self.Key])
            self.att_mem = F.softmax(self.att_key, dim=0)
            self.att_mem = torch.einsum('ab,abc->abc', [self.frien_num, self.att_mem])
            self.frien_f1 = torch.einsum('abc,ck->abk', [self.att_mem, self.Mem])
            self.frien_f2 = torch.mul(self.frien_f1, self.frien_embedding)
            # friend_attention
            self.frien_j = torch.exp(
                torch.einsum('abc,ck->abk', F.relu(torch.einsum('abc,ck->abk', self.frien_f2, self.WA) + self.BA),
                             self.U_omega))
            self.frien_j = torch.einsum('ab,abc->abc', [self.frien_num, self.frien_j])
            self.frien_sum = torch.sum(self.frien_j, dim=1).unsqueeze(1) + 1e-8
            self.frien_w = torch.div(self.frien_j, self.frien_sum)
            self.friend = torch.sum(torch.mul(self.frien_w, self.frien_f2), dim=1)
            self.friend = self.dropout(self.friend)
            self.user = self.uid + self.friend  # [B,H]

            #-----------------------------------------------------------------------
            att1 = torch.softmax(torch.matmul(torch.cat([self.uid, self.user],1),self.att_exp1),1)
            user_f = torch.mul(self.uid,att1[:,0].unsqueeze(1))+ torch.mul(self.user,att1[:,1].unsqueeze(1))
            #-----------------------------------------------------------------------

            self.score = torch.sum(torch.mul(user_f, self.iid), 1) + self.i_b
            if flag == 1: return self.score
            loss1 = self.criterion1(self.score, label)

        if (flag == 2) or (flag == 0):
            if flag == 0:
                inputs, mask, targets = data.get_slice(i)
            else:
                inputs, mask, targets, negs = data.get_slice(i)
                negs = torch.Tensor(negs).long().cuda()
            inputs = torch.Tensor(inputs).long().cuda()
            mask = torch.Tensor(mask).long().cuda()
            targets = torch.Tensor(targets).long().cuda()
            seq_l = torch.sum(mask, 1)

            # multi head
            mul_seq = torch.cat([att(self.uidW, inputs, seq_l) for att in self.in_att], dim=2).cuda()
            mul_seq_c = torch.cat([mul_seq[i] for i in range(mul_seq.size()[0])], dim=0)
            mul_one = torch.mm(mul_seq_c, self.w)
            mul_one = F.elu(mul_one)
            seq_hidden = self.out_att(self.uidW,
                                      mul_one.view(mul_seq.size()[0], mul_seq.size()[1], self.embedding_size),
                                      seq_l).cuda()
            scores = self.compute_scores(seq_hidden, inputs, mask)
            if (flag == 2): return scores, negs
            loss2 = self.criterion2(scores, targets)
        return loss1, loss2

    def compute_scores(self, hidden, inputs, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            p_a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.uidW[:-1]

        # ------------------------------
        p_i_all = self.uidW[inputs]
        p_i = p_i_all * mask.unsqueeze(2)
        p_maxpool = torch.max(p_i, dim=1)[0]
        
        att = torch.softmax(torch.matmul(torch.cat([p_a, p_maxpool], 1), self.att_t), 1)  # [B,2H] x [2H,2] = [B,2]
        a = p_a * att[:, 0].unsqueeze(1) + p_maxpool * att[:, 1].unsqueeze(1)
        # ------------------------------
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores
