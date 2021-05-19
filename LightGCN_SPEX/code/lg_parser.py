import argparse

def parse_args_r():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--cuda_id', default='0', help='which device to use')
    parser.add_argument('--data_path', nargs='?', default="../data/", help='Input data path.')
    parser.add_argument('--dataset', type=str, default='twitter', help="available datasets: [epinion2,weibo,twitter]")
    parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
    parser.add_argument('--recdim', type=int, default=64, help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3, help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--dropout', type=int, default=0, help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6, help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100, help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--A_split', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument('--hiddenSize', type=int, default=64, help='hidden state size : 100')
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
    parser.add_argument('--act', type=int, default=1,help='activation function')

    return parser.parse_args()