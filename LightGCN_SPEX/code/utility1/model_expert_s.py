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
        self.hidden_size = self.args_r.hiddenSize
        self.batch_size = self.args_r.batchSize
        self.nonhybrid = self.args_r.nonhybrid
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
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

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users + 1, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

        # multi_task
        self.task_weights = nn.Parameter(torch.FloatTensor([0.0, 0.0]))
        self.rec_loss = nn.BCEWithLogitsLoss()
        self.loss_function = nn.CrossEntropyLoss()

        # expert_s------------------------------------
        self.att_exp1 = nn.Parameter(torch.zeros(size=(2 * self.hidden_size, 2)))
        self.att_exp2 = nn.Parameter(torch.zeros(size=(2 * self.hidden_size, 2)))
        nn.init.xavier_uniform_(self.att_exp1.data, gain=1)
        nn.init.xavier_uniform_(self.att_exp2.data, gain=1)
        self.att_t = nn.Parameter(torch.zeros(size=(2 * self.hidden_size, 2)))
        nn.init.xavier_normal_(self.att_t.data, gain=1)
        # -----------------------------------------


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

    def compute_scores(self, hidden, inputs, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            p_a = self.linear_transform(torch.cat([a, ht], 1))
        
        b = self.embedding_user.weight[:-1]  # n_nodes x latent_size

        # ------------------------------
        p_i_all = self.embedding_user.weight[inputs]
        p_i = p_i_all * mask.unsqueeze(2)
        p_maxpool = torch.max(p_i, dim=1)[0]
        
        att = torch.softmax(torch.matmul(torch.cat([p_a, p_maxpool], 1), self.att_t), 1)  # [B,2H] x [2H,2] = [B,2]
        a = p_a * att[:, 0].unsqueeze(1) + p_maxpool * att[:, 1].unsqueeze(1)
        # ------------------------------
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores  

    def forward(self, users, items, labels, slice_indices, trust_data, flag=0):
        '''no tower two att'''
        # rec
        if (flag == 1) or (flag == 0):
            all_users_ex, all_items_ex = self.computer()
            #------------------------------
            all_users_or = self.embedding_user.weight
            all_items_or = self.embedding_item.weight
            att1 = torch.softmax(torch.matmul(torch.cat([all_users_or,all_users_ex],1),self.att_exp1),1)
            att2 = torch.softmax(torch.matmul(torch.cat([all_items_or,all_items_ex],1),self.att_exp2),1)
            all_users = torch.mul(all_users_or,att1[:,0].unsqueeze(1))+ torch.mul(all_users_ex,att1[:,1].unsqueeze(1))
            all_items = torch.mul(all_items_or,att2[:,0].unsqueeze(1))+ torch.mul(all_items_ex,att2[:,1].unsqueeze(1))
            # ------------------------------
            users_emb = all_users[users]
            items_emb = all_items[items]
            inner_pro = torch.mul(users_emb, items_emb)
            gamma = torch.sum(inner_pro, dim=1)
            if flag == 1: return gamma
            loss1 = self.rec_loss(gamma, labels.float())
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

            mul_seq = torch.cat([att(self.embedding_user.weight, inputs, seq_l) for att in self.in_att], dim=2).cuda()
            mul_seq_c = torch.cat([mul_seq[i] for i in range(mul_seq.size()[0])], dim=0)
            mul_one = torch.mm(mul_seq_c, self.w)
            mul_one = F.elu(mul_one)
            seq_hidden_att = self.out_att(self.embedding_user.weight,
                                          mul_one.view(mul_seq.size()[0], mul_seq.size()[1], self.hidden_size),
                                          seq_l).cuda()
            scores = self.compute_scores(seq_hidden_att, inputs, mask)
            if (flag == 2):
                return scores, negs
            loss2 = self.loss_function(scores, targets)
        return loss1, loss2
