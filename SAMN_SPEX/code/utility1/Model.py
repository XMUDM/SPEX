import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F


class SAMN_TRUST(nn.Module):

    def __init__(self, user_num, item_num, args):
        super(SAMN_TRUST, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = args.embedding_size

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
            self.user = self.uid + self.friend
            self.score = torch.sum(torch.mul(self.user, self.iid), 1) + self.i_b
            if flag == 1: return self.score
            loss1 = self.criterion1(self.score, label)
        return loss1

