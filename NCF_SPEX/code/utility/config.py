from ncf_parser import ncf_parse_args

args = ncf_parse_args()

# dataset name
dataset = args.dataset
# assert dataset in ['ml-1m', 'pinterest-20']

# model name 
model = 'NeuMF-end'
# assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = '../Data/'

train_rating = main_path + '{}/'.format(dataset) + 'rec/{}.train.rating'.format(dataset)
test_rating = main_path + '{}/'.format(dataset) + 'rec/{}.test.rating'.format(dataset)
test_negative = main_path + '{}/'.format(dataset) + 'rec/{}.test.negative'.format(dataset)

model_path = '../models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
