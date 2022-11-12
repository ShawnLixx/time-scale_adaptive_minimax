import random
import numpy as np
import torch

from utils.config import parse_args
from utils.data_loader import get_data_loader

from models.wgan_gradient_penalty import WGAN_GP


def main(args):
    model = WGAN_GP(args)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)

    # Start model training
    model.train(train_loader)


if __name__ == '__main__':
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)

    args = parse_args()
    print(args.cuda)
    main(args)
