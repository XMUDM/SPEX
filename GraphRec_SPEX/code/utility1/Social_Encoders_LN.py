import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Social_Encoder(nn.Module):

    def __init__(self, features, aggregator, embed_dim):
        super(Social_Encoder, self).__init__()

        self.features = features
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes):

        neigh_feats = self.aggregator.forward(nodes)
        self_feats = self.features.forward(nodes)

        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
