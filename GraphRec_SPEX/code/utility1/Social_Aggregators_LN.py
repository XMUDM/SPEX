import torch
import torch.nn as nn


class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, u2e, u_u, u_u_l):
        super(Social_Aggregator, self).__init__()

        self.u2e = u2e
        self.u_u = u_u
        self.u_u_l = u_u_l

    def forward(self, nodes):  
        tem_adj = self.u_u[nodes] 
        e_u = self.u2e.weight[tem_adj.long()] 

        N_a = self.u_u_l[nodes] 
        N_b = self.u_u_l[tem_adj.long()].squeeze(2)  
        att_w = (1/((N_a**(0.5))*(N_b**(0.5)))).unsqueeze(2) 
        att_w[att_w == float('inf')] = 0

        embed_matrix = torch.bmm(e_u.permute(0,2,1), att_w).squeeze(2) 

        return embed_matrix 
