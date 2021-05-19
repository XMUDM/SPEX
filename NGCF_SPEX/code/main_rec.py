import os
from ngcf_parser import parse_args
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
import random
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F

from utility.Logging import Logging
from utility.helper import *
from utility.batch_test import test, data_generator, args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(2020)

log_dir = os.path.join(os.getcwd(), 'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(os.getcwd(), 'log/%s_rec.log' % (args.dataset))
log = Logging(log_path)
log.record(args.act)

class Model_Wrapper(nn.Module):
    def __init__(self, data_config, device):
        super(Model_Wrapper, self).__init__()
        self.device = device
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.embedding_dim = args.embed_size

        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.norm_adj = data_config['norm_adj']
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float()
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.mess_dropout = eval(args.mess_dropout)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.dropout_list.append(nn.Dropout(self.mess_dropout[i]))

        self.user_embedding = nn.Embedding(self.n_users+1, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.rec_loss_function = nn.BCEWithLogitsLoss()


    def forward(self, user, item, labels_list, flag):
        if (flag == 0) or (flag == 1):
            ego_embeddings = torch.cat((self.user_embedding.weight[:-1], self.item_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_layers):
                side_embeddings = torch.sparse.mm(self.norm_adj.to(self.device), ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)
                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.cat(all_embeddings, dim=1)
            ua_embeddings, ia_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            if (flag == 1): return ua_embeddings, ia_embeddings

            u_g_embeddings = ua_embeddings[trans_to_cuda(user)]
            i_g_embeddings = ia_embeddings[trans_to_cuda(item)]
            rec_loss = self.compute_rec_loss(u_g_embeddings, i_g_embeddings, labels_list)

        return rec_loss


    def compute_rec_loss(self, u_g_embeddings, i_g_embeddings, labels_list):
        predict = torch.sum(torch.mul(u_g_embeddings, i_g_embeddings), dim=1)
        real_score = trans_to_cuda(labels_list)

        return self.rec_loss_function(predict, real_score)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_sparse_tensor_value(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape


def train(model, optimizer):
    best_recall, best_ndcg, best_iter = [0, 0, 0], [0, 0, 0], [0, 0]
    for epoch in range(args.epoch):
        t1 = time()
        total_loss = 0.
        data_loader = data_generator.load_train_data()
        for data in data_loader:
            model.train()
            optimizer.zero_grad()
            user, item, labels_list = data
            loss = model(user=trans_to_cuda(user), item=trans_to_cuda(item), labels_list=trans_to_cuda(labels_list), flag=0)
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()
        t2 = time()
        log.record('%d,%.5f,%.2f' % (epoch,total_loss,t2-t1))

        model.eval()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=True)
        perf_str = 'Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
            epoch, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1],
            ret['ndcg'][2])
        log.record(perf_str)
        if ret['recall'][0] > best_recall[0]:
            best_recall, best_iter[0] = ret['recall'], epoch
        if ret['ndcg'][0] > best_ndcg[0]:
            best_ndcg, best_iter[1] = ret['ndcg'], epoch

    log.record("--- Train Best ---")
    best_rec = 'Rec:  recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
    best_recall[0], best_recall[1], best_recall[2], best_ndcg[0], best_ndcg[1], best_ndcg[2])
    log.record(best_rec)


if __name__ == '__main__':

    data_generator.print_statistics()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    if args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        # print('use the normalized adjacency matrix')
    else:
        config['norm_adj'] = mean_adj + sparse.eye(mean_adj.shape[0])
        # print('use the mean adjacency matrix')

    Engine = Model_Wrapper(data_config=config, device=device).to(device)

    # multi
    optimizer = torch.optim.Adam(Engine.parameters(), lr=args.lr)
    train(Engine, optimizer)
