import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NCF.")
    parser.add_argument('--data_root', default='../data/', help='dataset root')
    parser.add_argument("--cuda_id", type=str, default="1", help="gpu card ID")
    parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
    parser.add_argument('--percesent', type=int, default=0,
                        help='new sample dataset percesent 100/80/50:新数据集, 0:就数据集')
    parser.add_argument("--dropout", type=float,default=0.0,  help="dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
    parser.add_argument("--epochs", type=int,default=50,  help="training epoches")
    parser.add_argument("--factor_num", type=int,default=64,  help="predictive factors numbers in the model")
    parser.add_argument("--num_layers", type=int,default=3, help="number of layers in MLP model")
    parser.add_argument("--num_ng", type=int,default=5, help="sample negative items for training")
    parser.add_argument('--dataset', default='epinion2', help='dataset name: epinion2/twitter/weibo')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden state size : 100')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
    return parser.parse_args()