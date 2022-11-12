import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

    parser.add_argument('--dataroot', default='dataset', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'],
                            help='The name of dataset')
    parser.add_argument('--download', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=512, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=50, help='latent variable dimension')
    parser.add_argument('--cuda',  type=str, default='True', help='Availability of cuda')
    parser.add_argument('--optim',  type=str, default='adam', help='optimizer to use')
    parser.add_argument('--lr',  type=float, default=1e-4, help='learning rate')
    parser.add_argument('--critic_iter',  type=int, default=5, help='number of critic iteration')
    parser.add_argument('--alpha',  type=float, default=0.6, help='parameter in TiAda')
    parser.add_argument('--beta',  type=float, default=0.4, help='parameter in TiAda')

    parser.add_argument('--generator_iters', type=int, default=40000, help='The number of iterations for generator in WGAN model.')
    return check_args(parser.parse_args())


# Checking arguments
def check_args(args):
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    if args.dataset == 'cifar' or args.dataset == 'stl10':
        args.channels = 3
    else:
        args.channels = 1
    args.cuda = True if args.cuda == 'True' else False
    return args
