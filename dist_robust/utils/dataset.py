import os

import torch
import torch.utils.data
import torchvision
import numpy as np

from config import args

def get_dataset():
  if args.dataset == 'mnist':
    # Configure data loader
    os.makedirs("./dataset/mnist", exist_ok=True)
    mnist_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=mnist_transforms, download=True)
    indices = list(range(len(mnist_train)))
    np.random.shuffle(indices)
    TRAIN_DATA_SIZE = 50000
    train_idx, test_idx = indices[:TRAIN_DATA_SIZE], indices[TRAIN_DATA_SIZE:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    train_data_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=args.batchsize, sampler=train_sampler, num_workers=10)
    test_data_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=args.batchsize,  sampler=test_sampler, num_workers=10)
  elif args.dataset == 'synthetic':
    raise NotImplementedError
  else:
    raise NotImplementedError

  return train_data_loader, test_data_loader

def sample_latent(shape):
  return torch.randn(shape)
