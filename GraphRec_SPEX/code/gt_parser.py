import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--cuda_id', default='1', help='which gpu to use.')#
    parser.add_argument('--dataset', default='epinion2', help='dataset name: epinion2\weibo\twitter')#
    parser.add_argument('--percesent', type=int, default=0, help='new sample dataset percesent 100/80/50:新数据集, 0:就数据集')
    parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')#
    
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',help='input batch size for training')#
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N',help='embedding size')#
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate')#
    parser.add_argument('--epochs', type=int, default=50, metavar='N',help='number of epochs to train')#
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')#
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')#
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]#
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')#
    
    return parser.parse_args()