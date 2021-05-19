import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run MyModel.")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epoch.')  # 450
    parser.add_argument('--cuda_id', type=str, default='0', help='cuda_id')
    parser.add_argument('--data_root', default='../data/', help='dataset root')
    parser.add_argument('--dataset', default='twitter', help='dataset name')
    parser.add_argument('--links_filename', default='social.share', help='u1 trust u2 filename')
    parser.add_argument('--model_name', type=str, default='diffnetplus',help='model_name.')
    parser.add_argument('--num_negatives', type=int, default=5,help='num_negatives.')
    parser.add_argument('--num_procs', type=int, default=16, help='num_procs.')
    parser.add_argument('--num_evaluate', type=int, default=100, help='num_evaluate.')
    parser.add_argument('--batch_size', type=int, default=256, help='training_batch_size.')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden state size : 100')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--pretrain_flag', type=int, default=0, help='if pretrain.')
    parser.add_argument('--pre_model', type=str, default='string diffnet_hr_0.3437_ndcg_0.2092_epoch_98.ckpt', help='pertain model.')
    # trust
    parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
    return parser.parse_args()




