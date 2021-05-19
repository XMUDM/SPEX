import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='DANSER model')
    parser.add_argument('--data_root', default='../data/', help='dataset root')
    parser.add_argument('--dataset', default='epinion2', help='dataset name: epinion2, weibo, twitter')

    parser.add_argument('--embedding_size', type=int, default=16, metavar='N',help='embedding size') 
    parser.add_argument('--train_batch_size', type=int, default=32, metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N', help='input batch size for testing')
    parser.add_argument('--trunc_len', type=int, default=10, metavar='N', help='number of trunc_len')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train')
    parser.add_argument('--cuda_id', default='0', help='which device to use')
    parser.add_argument("--local_rank", type=int, help="")

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--keep_prob', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--Ks', type=int, default=[10,20,50], help='metrics to k')

    parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')

    return parser.parse_args()