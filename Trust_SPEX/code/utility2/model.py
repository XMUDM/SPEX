import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility2.gpuutil import trans_to_cuda
from utility2.layers import GraphAttentionLayer


class TRUST_PATH(nn.Module):
    def __init__(self, opt):
        super(TRUST_PATH, self).__init__()

        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size
        self.nonhybrid = opt.nonhybrid
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.reset_parameters()

        # multi head
        self.in_att = [GraphAttentionLayer(opt.hidden_size, concat=True) for _ in range(opt.nb_heads)]
        for i, attention in enumerate(self.in_att):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(opt.hidden_size, concat=False)
        self.w = nn.Parameter(torch.zeros(size=(opt.nb_heads * self.hidden_size, self.hidden_size)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)

        self.trust_user_embs = nn.Embedding(opt.user_num + 1, self.hidden_size)
        nn.init.xavier_uniform_(self.trust_user_embs.weight, gain=1)

        # expert_s------------------------------------
        self.att_t = nn.Parameter(torch.zeros(size=(2 * self.hidden_size, 2)))
        nn.init.xavier_uniform_(self.att_t.data, gain=1)
        # -----------------------------------------

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, user, item, i, data, flag):
        if flag == 0:
            inputs, mask, targets = data.get_slice(i)
        else:
            inputs, mask, targets, negs = data.get_slice(i)
            negs = trans_to_cuda(torch.Tensor(negs).long())

        inputs = trans_to_cuda(torch.Tensor(inputs).long())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        targets = trans_to_cuda(torch.Tensor(targets).long())
        seq_l = torch.sum(mask, 1)

        # multi head
        mul_seq = trans_to_cuda(
            torch.cat([att(self.trust_user_embs.weight, inputs, seq_l) for att in self.in_att], dim=2))
        mul_seq_c = torch.cat([mul_seq[i] for i in range(mul_seq.size()[0])], dim=0)
        mul_one = torch.mm(mul_seq_c, self.w)
        mul_one = F.elu(mul_one)
        seq_hidden = trans_to_cuda(self.out_att(self.trust_user_embs.weight,
                                                mul_one.view(mul_seq.size()[0], mul_seq.size()[1], self.hidden_size),
                                                seq_l))

        trust_scores = self.compute_trust_scores(seq_hidden, inputs, mask)
        if flag == 0:
            return trust_scores, targets
        else:
            return trust_scores, negs

    def compute_trust_scores(self, hidden, inputs, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        p_a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            p_a = self.linear_transform(torch.cat([p_a, ht], 1))
        b = self.trust_user_embs.weight[:-1]  # n_nodes x latent_size

        # ------------------------------
        p_i_all = self.trust_user_embs.weight[inputs]
        p_i = p_i_all * mask.unsqueeze(2)
        p_maxpool = torch.max(p_i, dim=1)[0]

        att = torch.softmax(torch.matmul(torch.cat([p_a, p_maxpool], 1), self.att_t), 1)  # [B,2H] x [2H,2] = [B,2]
        a = p_a * att[:, 0].unsqueeze(1) + p_maxpool * att[:, 1].unsqueeze(1)
        # ------------------------------

        scores = torch.matmul(a, b.transpose(1, 0))
        return scores
