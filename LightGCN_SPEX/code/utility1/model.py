import math
import torch
from torch import nn
import torch.nn.functional as F
from utility1.dataloader import BasicDataset
from utility1.gpuutil import trans_to_cpu, trans_to_cuda



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

        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.args_r.recdim
        self.n_layers = self.args_r.layer
        self.keep_prob = self.args_r.keepprob
        self.A_split = self.args_r.A_split

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users + 1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        # print(f"lgn is already to go(dropout:{self.args_r.dropout})")

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
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.args_r.dropout:  # 0
            if self.training:
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
        b = self.embedding_user.weight[:-1]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores  

    def forward(self, users, items, labels, flag=0):
        # rec
        if (flag == 1) or (flag == 0):
            all_users, all_items = self.computer()
            users_emb = all_users[users]
            items_emb = all_items[items]
            inner_pro = torch.mul(users_emb, items_emb)
            gamma = torch.sum(inner_pro, dim=1)
            if flag == 1: return gamma
            loss = self.bcel(gamma, labels.float())
        return loss
