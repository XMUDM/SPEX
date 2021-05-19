import torch.nn as nn
import torch.nn.functional as F
import torch

class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_size, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.concat = concat
        self.hidden_size = hidden_size

        self.a = nn.Parameter(torch.zeros(size=(2*hidden_size, 1)))  # [128, 1]
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(inplace=True)

    def forward(self,emb,seq,seq_l):
        if self.concat:
            at_hidden = torch.ones((seq.size()[0],seq.size()[1],self.hidden_size))  # [135, 5, 64]
            for p in range(len(seq)):
                for i in range(int(seq_l[p].item()-1)):

                    ue_pos_emb = torch.full([self.hidden_size], seq_l[p].item()-i, dtype=torch.float).cuda()
                    fe_pos_emb = torch.full([self.hidden_size], seq_l[p].item()-i-1, dtype=torch.float).cuda()

                    # no position embedding
                    # ue = emb(seq[p][i].long()).repeat(2, 1)
                    # fe = torch.stack([emb(seq[p][i].long()),emb(seq[p][i+1].long())], dim=0)

                    # add position embedding
                    ue = (emb[seq[p][i].long()]+ue_pos_emb).repeat(2, 1)
                    fe = torch.stack([(emb[seq[p][i].long()]+ue_pos_emb), (emb[seq[p][i + 1].long()]+fe_pos_emb)], dim=0)

                    h = torch.cat([ue,fe],dim=1)  # [2, 128]
                    att = torch.mm(h,self.a).squeeze(1) # [2]
                    att = F.softmax(att, dim=0)
                    u = torch.matmul(att.t(),fe)  # [128, 64]
                    at_hidden[p][i] = u
                for i in range(int(seq_l[p].item()-1),seq.size()[1]):
                    ue = emb[seq[p][i].long()]
                    at_hidden[p][i] = ue
            return at_hidden
        else:
            at_hidden = torch.ones((seq.size()[0],seq.size()[1],self.hidden_size))
            for p in range(len(seq)):
                for i in range(int(seq_l[p].item()-1)):
                    ue = seq[p][i].repeat(2, 1)
                    fe = torch.stack([seq[p][i],seq[p][i+1]], dim=0)
                    h = torch.cat([ue,fe],dim=1)
                    att = torch.mm(h,self.a).squeeze(1)
                    att = F.softmax(att, dim=0)
                    u = torch.matmul(att.t(),fe)
                    at_hidden[p][i] = u
                for i in range(int(seq_l[p].item()-1),seq.size()[1]):
                    ue = seq[p][i]
                    at_hidden[p][i] = ue
            return at_hidden

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
