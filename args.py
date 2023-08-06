import time
import torch
import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="testrun", help='Provide a test name.')
'''
python run.py --dataset texas --Ts 4.0 --topk 10 --lambd 0.8
python run.py --dataset cornell --Ts 1.0 --topk 5 --lambd 0.9
python run.py --dataset wisconsin --Ts 2.0 --topk 15 --lambd 0.9
python run.py --dataset chameleon --Ts 1.0 --topk 0 --lambd 0.7
python run.py --dataset cora --Ts 5.0 --topk 20 --lambd 0.7
python run.py --dataset citeseer --Ts 5.0 --topk 0 --lambd 0.9
python run.py --dataset squirrel --Ts 2.0 --topk 5 --lambd 0.4
python run.py --dataset pubmed --Ts 1.0 --topk 0 --lambd 0.3
'''

#Feature Teacher
parser.add_argument('--num_fea_layers', type=int, default=2, help='Number pf layers for Feature Teacher')
parser.add_argument('--hidden_fea', type=int, default=64,help='Number of hidden units.')
parser.add_argument('--dropout_fea', type=float, default=0, help='Dropout rate.')
parser.add_argument('--epoch_fea', type=int, default=100, help='Number of epochs for Feature Teacher')
parser.add_argument('--lr_fea', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay_fea', type=float, default=0.0005, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--batch_size_fea', type=int, default=1024, help='batch_size seed.')

#Structure Teacher
#Student (GCN)
parser.add_argument('--epoch_stu', type=int, default=100, help='Max number of epochs for gcn. Default is 400.')
parser.add_argument('--num_gcn_layers', type=int, default=2, help='Number pf layers for gcn')
parser.add_argument('--hidden_stu', type=int, default=64,help='Number of hidden units.')
parser.add_argument('--dropout_stu', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--Ts', type=float, default=1.0, help='temperature for ST')
parser.add_argument('--lambd', type=float, default=0.0, help='trade-off parameter for kd loss')
parser.add_argument('--lr_stu', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay_stu', type=float, default=5e-4,  help='Weight decay (L2 loss on parameters).')
parser.add_argument('--eps',type=float, default=0.5,  help='Weight decay (L2 loss on parameters).')
parser.add_argument('--beta1', type=float, default=1, help='Weight balance1')
parser.add_argument('--beta2', type=float, default=1, help='Weight balance2')


parser.add_argument('--dataset', type=str, default="yelp", help='dataset.')
parser.add_argument('--rate', default=0.8, type=float, help='masking rate')

parser.add_argument('--train_rate',default=0.4, type=float, help='training rate')
parser.add_argument('--test_rate',default=0.67, type=float, help='training rate')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--repeat', type=int, default=1, help='repeat.')
parser.add_argument('--no_cuda', action='store_false', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
args = parser.parse_args()
args.device = torch.device('cpu' if args.no_cuda and torch.cuda.is_available() else 'cpu')
args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

