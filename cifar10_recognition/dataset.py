import numpy as np
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit


def get_dataset(dataset, date_path, batch_size):
    if dataset.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dst = CIFAR10(root=date_path, train=True, download=True, transform=transform_train)
        test_dst = CIFAR10(root=date_path, train=False, download=True, transform=transform_test)

        labels = [0 for _ in range(len(train_dst))]
        ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]

        train_queue = DataLoader(
            torch.utils.data.Subset(train_dst, train_indices),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        valid_queue = DataLoader(
            torch.utils.data.Subset(train_dst, valid_indices),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        test_queue = DataLoader(
            test_dst,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
    else:
        raise Exception('Invalid Dataset.')

    return train_queue, valid_queue, test_queue
