import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='SAMN model')
    parser.add_argument('--data_root', default='../data/', help='dataset root')
    parser.add_argument('--dataset', default='twitter', help='dataset name: epinion2, weibo, twitter')

    parser.add_argument('--embedding_size', type=int, default=64, metavar='N',help='embedding size')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train')
    parser.add_argument('--cuda_id', default='1', help='which device to use')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N', help='input batch size for testing')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--dp', type=float, default=1.0, help='dropout rate')
    parser.add_argument('--memory_size', type=int, default=8, help='memory_size.')
    parser.add_argument('--attention_size', type=int, default=16, help='attention_size.')
    parser.add_argument('--Ks', type=int, default=[10,20,50], help='metrics to k')

    parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')

    return parser.parse_args()
