import torch
import torch.nn as nn


class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, u2e, u_v, v_u, u_v_l, v_u_l, uv=True):
        super(UV_Aggregator, self).__init__()
        
        self.v2e = v2e
        self.u2e = u2e
        self.uv = uv
        self.u_v = u_v
        self.v_u = v_u
        self.u_v_l = u_v_l
        self.v_u_l = v_u_l
        self.uv = uv

    def forward(self, nodes):
        # user component
        if self.uv == True:
            history = self.u_v[nodes]  
            e_v = self.v2e.weight[history.long()] 
            N_a = self.u_v_l[nodes] 
            N_b = self.v_u_l[history.long()].squeeze(2) 
            att_w = (1/((N_a**(0.5))*(N_b**(0.5)))).unsqueeze(2) 
            att_w[att_w == float('inf')] = 0
            embed_matrix = torch.bmm(e_v.permute(0,2,1), att_w).squeeze(2)
        else:
            # item component
            history = self.v_u[nodes]  
            e_u = self.u2e.weight[history.long()] 

            N_a = self.v_u_l[nodes] 
            N_b = self.u_v_l[history.long()].squeeze(2)  
            att_w = (1/((N_a**(0.5))*(N_b**(0.5)))).unsqueeze(2) 
            att_w[att_w == float('inf')] = 0
            embed_matrix = torch.bmm(e_u.permute(0,2,1), att_w).squeeze(2)
  
        return embed_matrix
