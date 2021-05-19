import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility2.layers import GraphAttentionLayer
from utility.gpuutil import trans_to_cuda


class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout, model, opt):
        super(NCF, self).__init__()
        # trust
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size
        self.nonhybrid = opt.nonhybrid
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        # multi task
        self.rec2trust_useremb_mlp = nn.Sequential()
        self.rec2trust_useremb_mlp.add_module("Linear_layer_1", nn.Linear(5 * factor_num, 2 * factor_num))
        self.rec2trust_useremb_mlp.add_module("Relu_layer_1", nn.ReLU(inplace=True))
        self.rec2trust_useremb_mlp.add_module("Linear_layer_2", nn.Linear(2 * factor_num, factor_num))
        self.rec2trust_useremb_mlp.add_module("Relu_layer_2", nn.ReLU(inplace=True))

        self.reset_parameters()

        # multi head
        self.in_att = [GraphAttentionLayer(opt.hidden_size, concat=True) for _ in range(opt.nb_heads)]
        for i, attention in enumerate(self.in_att):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(opt.hidden_size, concat=False)
        self.w = nn.Parameter(torch.zeros(size=(opt.nb_heads * self.hidden_size, self.hidden_size)))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)

        # rec
        self.dropout = dropout
        self.model = model
        self.embed_user_GMF = nn.Embedding(user_num + 1, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num + 1, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(item_num, factor_num * (2 ** (num_layers - 1)))
        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)
        self.init_weight()

        self.log_vars = nn.Parameter(torch.zeros((2), requires_grad=True))

        # expert_sr------------------------------------
        self.att_exp1 = nn.Parameter(torch.zeros(size=(2 * self.hidden_size, 2)))
        self.att_exp2 = nn.Parameter(torch.zeros(size=(2 * self.hidden_size, 2)))
        nn.init.xavier_uniform_(self.att_exp1.data, gain=1)
        nn.init.xavier_uniform_(self.att_exp2.data, gain=1)
        # expert_st------------------------------------
        self.att_t = nn.Parameter(torch.zeros(size=(2 * self.hidden_size, 2)))
        nn.init.xavier_uniform_(self.att_t.data, gain=1)
        # -----------------------------------------

    def reset_parameters(self):
       stdv = 1.0 / math.sqrt(self.hidden_size)
       for weight in self.parameters():
           weight.data.uniform_(-stdv, stdv)

    def init_weight(self):
        """ We leave the weights initialization here. """
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight,
                                     a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, user, item, slice_indices, trust_data, flag):

        if (flag == 0) or (flag == 1):
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)

            output_MLP = self.MLP_layers(interaction)
            concat = torch.cat((output_GMF, output_MLP), -1)
            rec_prediction = self.predict_layer(concat).view(-1)

            if (flag == 1): return rec_prediction

        if (flag == 0) or (flag == 2):
            if flag == 0:
                inputs, mask, targets = trust_data.get_slice(slice_indices)
            else:
                inputs, mask, targets, negs = trust_data.get_slice(slice_indices)
                negs = trans_to_cuda(torch.Tensor(negs).long())

            inputs = trans_to_cuda(torch.Tensor(inputs).long())
            mask = trans_to_cuda(torch.Tensor(mask).long())
            targets = trans_to_cuda(torch.Tensor(targets).long())
            seq_l = torch.sum(mask, 1) 

            self.trust_user_embs = self.rec2trust_useremb_mlp(torch.cat([self.embed_user_GMF.weight, self.embed_user_MLP.weight], dim=1))

            # multi head
            mul_seq = trans_to_cuda(torch.cat([att(self.trust_user_embs, inputs, seq_l) for att in self.in_att], dim=2))
            mul_seq_c = torch.cat([mul_seq[i] for i in range(mul_seq.size()[0])], dim=0)
            mul_one = torch.mm(mul_seq_c, self.w)
            mul_one = F.elu(mul_one)
            seq_hidden = trans_to_cuda(
                self.out_att(self.trust_user_embs, mul_one.view(mul_seq.size()[0], mul_seq.size()[1], self.hidden_size), seq_l))

            trust_scores = self.compute_scores(seq_hidden, inputs, mask)
            if (flag == 2): return trust_scores, negs

        return rec_prediction, trust_scores, targets

    def compute_scores(self, hidden, inputs, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        p_a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            p_a = self.linear_transform(torch.cat([p_a, ht], 1))
        b = self.trust_user_embs[:-1]  # n_nodes x latent_size
        # ------------------------------
        p_i_all = self.trust_user_embs[inputs]
        p_i = p_i_all * mask.unsqueeze(2)
        p_avgpool = torch.sum(p_i, dim=1) / torch.sum(mask, dim=1).unsqueeze(1)

        att = torch.softmax(torch.matmul(torch.cat([p_a, p_avgpool], 1), self.att_t), 1)  # [B,2H] x [2H,2] = [B,2]
        a = p_a * att[:, 0].unsqueeze(1) + p_avgpool * att[:, 1].unsqueeze(1)
        # ------------------------------
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores
