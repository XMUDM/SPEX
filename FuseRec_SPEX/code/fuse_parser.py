import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--cuda_id', default='0', help='which device to use')
    parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--dropout', type=int, default=0, help="using the dropout or not")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument('--hiddenSize', type=int, default=64, help='hidden state size : 100')
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')

    return parser.parse_args()