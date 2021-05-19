import os
from fuse_parser import parse_args
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
filename ='log/epinion2_rec2_0.001_clip0.008.log'
print(filename)
import random
import numpy as np
import torch
from time import time
from torch.utils.data import DataLoader
from utility.Input import Dataset
from utility.Model import FuseRec
from utility.metrics import rec_test
from utility.Logging import Logging

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(args.seed)

log_dir = os.path.join(os.getcwd(), 'log')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(os.getcwd(),filename)
log = Logging(log_path)

# load data
dataset = Dataset("epinion2")
rec_train_loader = DataLoader(dataset=dataset.generate_train_data(), batch_size=args.batchSize, shuffle=True)
rec_test_loader = DataLoader(dataset=dataset.generate_test_data(), batch_size=100, shuffle=False)

if __name__ == "__main__":
    model = FuseRec(dataset.user_num, dataset.item_num, args.hiddenSize, dataset.load_type()).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  
    bestres = [0, 0, 0, 0, 0, 0]
    for epoch in range(args.epochs):
        # train
        t1 = time()
        model.train()
        running_loss = 0.0
        for data in rec_train_loader:
            user, item, label, u_items, u_items_mask, u_frids, u_frids_mask, u_frids_items, F_i = data
            optimizer.zero_grad()
            loss1 = model(user.cuda(), item.cuda(), label.cuda(), u_items.cuda(), u_items_mask.cuda(), u_frids.cuda(), u_frids_mask.cuda(), u_frids_items.cuda(), F_i.cuda(), 0)
            loss_reg = model.reg_loss()
            loss = loss1 + 0.001*loss_reg  

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.008, norm_type=2)
            optimizer.step()
            running_loss += loss.item()
        t2 = time()
        train_rec = 'Epoch [%d], Loss [%.5f], Time[%.4f]' % (epoch, running_loss, t2-t1)
        log.record(train_rec)

        # test
        model.eval()
        ret = rec_test(model, rec_test_loader, dataset.test_len / 100)
        for i in range(len(bestres)):
            bestres[i] = max(bestres[i], ret[i])
        t3 = time()
        perf_str = 'Rec:  Epoch %d : recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f], Time=[%.4f]' % (
        epoch, ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], t3-t2)
        log.record(perf_str)

    perf_str = 'RecBest: recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (
    bestres[0], bestres[1], bestres[2], bestres[3], bestres[4], bestres[5])
    log.record(perf_str)
