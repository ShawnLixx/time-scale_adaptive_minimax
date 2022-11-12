import argparse

parser = argparse.ArgumentParser()

# training ralated
parser.add_argument('--num_epoch', type=int, default=200, help='number of epochs')
parser.add_argument('--batchsize', type=int, default=128, help='batch size')
parser.add_argument('--dataset', type=str, default='mnist', help='choose dataset')
parser.add_argument('--gamma', type=float, default=1.3, help='gamma in the loss')
parser.add_argument('--epsilon', type=float, default=0.1, help='attack level')

parser.add_argument('--model', type=str, default='CNN', help='model used')
# Optimizer
parser.add_argument('--optim', type=str, default='adam', help='optimization method')
parser.add_argument('--lr_x', type=float, default=0.0002, help='learning rate for g')
parser.add_argument('--lr_y', type=float, default=0.0002, help='learning rate for c')

parser.add_argument('--alpha', type=float, default=0.6, help='parameter for TiAda')
parser.add_argument('--beta', type=float, default=0.4, help='parameter for TiAda')
parser.add_argument('--n_inner', type=int, default=15, help='number of inner loop steps')

# directory related
parser.add_argument('--model_name', type=str, default=None, help='sub-directory name')

# others
parser.add_argument('--seed', type=int, default=8, help='random seed for reproducibility')

# parse
args = parser.parse_args()
