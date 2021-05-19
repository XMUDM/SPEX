import math
import torch
from torch import nn
import torch.nn.functional as F
from utility1.dataloader import BasicDataset
from utility1.gpuutil import trans_to_cpu, trans_to_cuda
from utility2.layers import GraphAttentionLayer

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self, args_r, dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.args_r = args_r
        self.dataset: BasicDataset = dataset
        self.bcel = nn.BCEWithLogitsLoss()
        # trust2
        self.hidden_size = self.args_r.hiddenSize
        self.batch_size = self.args_r.batchSize
        self.nonhybrid = self.args_r.nonhybrid
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()  # 损失函数

        # share----------------------------------------------------------------------
        self.mlp = nn.Sequential()
        self.mlp.add_module("Linear_layer_s", nn.Linear(self.hidden_size, self.hidden_size))
        self.mlp.add_module("Relu_layer_s", nn.ReLU(inplace=True))

        self.mlp_r = nn.Sequential()
        self.mlp_r.add_module("Linear_layer_r", nn.Linear(self.hidden_size * 2, self.hidden_size))
        self.mlp_r.add_module("Relu_layer_r", nn.ReLU(inplace=True))

        self.mlp_t = nn.Sequential()
        self.mlp_t.add_module("Linear_layer_t", nn.Linear(self.hidden_size * 2, self.hidden_size))
        self.mlp_t.add_module("Relu_layer_t", nn.ReLU(inplace=True))
        #---------------------------------------------------------------------
        self.reset_parameters()

        # multi head
        self.in_att = [GraphAttentionLayer(self.hidden_size, concat=True) for _ in range(self.args_r.nb_heads)]
        for i, attention in enumerate(self.in_att):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(self.hidden_size, concat=False)
        self.w = nn.Parameter(torch.zeros(size=(self.args_r.nb_heads * self.hidden_size, self.hidden_size)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)

        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.args_r.recdim
        self.n_layers = self.args_r.layer
        self.keep_prob = self.args_r.keepprob
        self.A_split = self.args_r.A_split

        self.u2e_r = nn.Embedding(self.num_users + 1, self.latent_dim)
        self.u2e_t = nn.Embedding(self.num_users + 1, self.latent_dim)
        self.embedding_item = torch.nn.Embedding(self.num_items, self.latent_dim)
        nn.init.xavier_uniform_(self.u2e_r.weight, gain=1)
        nn.init.xavier_uniform_(self.u2e_t.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

        # multi_task
        self.task_weights = nn.Parameter(torch.FloatTensor([0.0, 0.0]))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.u2e_r_new.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.args_r.dropout:  # 0
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:  # false
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users + 1, self.num_items])
        return users, items

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.u2e_t.weight[:-1]
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, users, items, labels, slice_indices, trust_data, flag=0):
        share_u_r = self.mlp(self.u2e_r.weight)
        share_u_t = self.mlp(self.u2e_t.weight)
        self.u2e_r_new = nn.Parameter(self.mlp_r(torch.cat([self.u2e_r.weight, share_u_r], dim=1)))
        self.u2e_t_new = nn.Parameter(self.mlp_t(torch.cat([self.u2e_t.weight, share_u_t], dim=1)))
        # rec
        if (flag == 1) or (flag == 0):
            all_users, all_items = self.computer()
            users_emb = all_users[users]
            items_emb = all_items[items]
            inner_pro = torch.mul(users_emb, items_emb)
            gamma = torch.sum(inner_pro, dim=1)
            if flag == 1: return gamma
            loss1 = self.bcel(gamma, labels.float())
            # trust
        if (flag == 2) or (flag == 0):
            if flag == 0:
                inputs, mask, targets = trust_data.get_slice(slice_indices)
            else:
                inputs, mask, targets, negs = trust_data.get_slice(slice_indices)
                negs = trans_to_cuda(torch.Tensor(negs).long())

            inputs = trans_to_cuda(torch.Tensor(inputs).long())
            mask = trans_to_cuda(torch.Tensor(mask).long())
            targets = trans_to_cuda(torch.Tensor(targets).long())
            seq_l = torch.sum(mask, 1) 
            # multi head
            mul_seq = torch.cat([att(self.u2e_t_new.weight, inputs, seq_l) for att in self.in_att], dim=2).cuda()
            mul_seq_c = torch.cat([mul_seq[i] for i in range(mul_seq.size()[0])], dim=0)
            mul_one = torch.mm(mul_seq_c, self.w)
            mul_one = F.elu(mul_one)
            seq_hidden_att = self.out_att(self.u2e_t_new.weight,
                                          mul_one.view(mul_seq.size()[0], mul_seq.size()[1], self.hidden_size),
                                          seq_l).cuda()
            scores = self.compute_scores(seq_hidden_att, mask)
            if (flag == 2): return scores, negs
            loss2 = self.loss_function(scores, targets)
        return loss1, loss2
