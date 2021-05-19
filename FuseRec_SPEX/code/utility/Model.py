import torch
import torch.nn as nn
import torch.nn.functional as F


class FuseRec(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, i_class_list):
        super(FuseRec, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        i_class_num = len(set(i_class_list))
        self.i_class = F.one_hot(torch.tensor(i_class_list).long(), i_class_num).float().cuda()  
        self.k = 10

        self.user_embedding = nn.Embedding(n_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim)
        self._init_weight_()

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

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def reg_loss(self):
        return torch.norm(input=self.user_embedding.weight, p=2)+torch.norm(input=self.item_embedding.weight, p=2)

    def forward(self, users, items, labels, u_items, u_items_mask, u_frids, u_frids_mask, u_frids_items, F_i, flag=0):

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
            yi = self.alpha * torch.sum(bij_1 * Fi_1, 1) + (1 - self.alpha) * torch.sum(Fi_2 * bij_2, 1)  # [B,D]

            # Compute score
            S1 = torch.sum(hu * yi, 1)  # inner product # [B,]
            S2 = torch.sum(hui * yi, 1)
            S3 = torch.sum(su * yi, 1)
            S4 = torch.sum(sui * yi, 1)
            rui = F.sigmoid(self.lambdas[0] * S1 + self.lambdas[1] * S2 + self.lambdas[2] * S3 + self.lambdas[3] * S4)

            if flag == 1: return rui

        return self.rec_loss(rui, labels.float())
