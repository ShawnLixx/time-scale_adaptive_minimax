import os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from utils.fashion_mnist import MNIST, FashionMNIST


def get_data_loader(args):

    dataroot = os.path.join(args.dataroot, args.dataset)
    if args.dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = MNIST(root=dataroot, train=True, download=args.download, transform=trans)
        test_dataset = MNIST(root=dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = FashionMNIST(root=dataroot, train=True, download=args.download, transform=trans)
        test_dataset = FashionMNIST(root=dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.CIFAR10(root=dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.CIFAR10(root=dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'stl10':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = dset.STL10(root=dataroot, split='train', download=args.download, transform=trans)
        test_dataset = dset.STL10(root=dataroot,  split='test', download=args.download, transform=trans)

    # Check if everything is ok with loading datasets
    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader
