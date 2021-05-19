import os
from lg_parser import parse_args_r
args = parse_args_r()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
import time
import torch
from torch.utils.data import DataLoader
from utility1.dataloader import LightTrainData
import utility1.dataloader as dataloader
import utility1.utils as utils
import utility1.model as model
from utility1.batch_test import test


utils.set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = dataloader.Loader(args)
train_dataset = LightTrainData(dataset.rec_train_data, dataset.m_item, dataset.train_mat)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

Recmodel = model.LightGCN(args, dataset).to(device)
optimizer = torch.optim.Adam(Recmodel.parameters(), lr=args.lr)

def Train(train_loader, recommend_model, epoch, optimizer):
    train_loader.dataset.ng_sample()
    Recmodel = recommend_model
    Recmodel.train()
    total_loss = 0.0
    for data in train_loader:
        # rec
        optimizer.zero_grad()
        user, item, label = data
        loss = Recmodel(users=user.to(device), items=item.to(device), labels=label.to(device), flag=0)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print('%d,%.5f' % (epoch, total_loss))

def Test(dataset, Recmodel, epoch, best_recall, best_ndcg, best_iter):
    # rec
    dataset: utils.BasicDataset
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    testRatings = dataset.testRatings
    testNegatives = dataset.testNegatives
    with torch.no_grad():
        # rec
        ret = test(Recmodel, testRatings, testNegatives)
        perf_str = 'Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
            epoch, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['ndcg'][0], ret['ndcg'][1],
            ret['ndcg'][2])
        print(perf_str)
        if ret['recall'][0] > best_recall[0]:
            best_recall, best_iter[0] = ret['recall'], epoch
        if ret['ndcg'][0] > best_ndcg[0]:
            best_ndcg, best_iter[1] = ret['ndcg'], epoch

    return best_recall, best_ndcg, best_iter


if __name__ == "__main__":
    best_recall, best_ndcg, best_iter = [0, 0, 0], [0, 0, 0], [0, 0]
    for epoch in range(args.epochs):
        start = time.time()
        Train(train_loader, Recmodel, epoch, optimizer)
        best_recall, best_ndcg, best_iter = Test(dataset, Recmodel, epoch, best_recall, best_ndcg, best_iter)
        end = time.time()
        train_time = end - start

    print("--- Train Best ---")
    best_rec = 'Rec:  recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (best_recall[0], best_recall[1], best_recall[2], best_ndcg[0], best_ndcg[1],best_ndcg[2])
    print(best_rec)
