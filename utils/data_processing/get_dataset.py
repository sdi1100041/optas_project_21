import torch
import numpy as np
from torchvision import datasets, transforms

CIFAR_TRAIN_DATASET_SIZE=50000
SUBSET_SIZE=25000

class CIFAR10Transform:
    def train_transform():
        return transforms.Compose([
               transforms.RandomCrop(32, padding=4),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    def test_transform():
        return transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

def get_train_data():
    dataset= datasets.CIFAR10(root='../data', train=True, download=True, transform=CIFAR10Transform.train_transform())
    dataset1=torch.utils.data.Subset(dataset,range(SUBSET_SIZE))
    dataset2=torch.utils.data.Subset(dataset,range(SUBSET_SIZE,CIFAR_TRAIN_DATASET_SIZE))

    dataset.targets = np.array(dataset.targets)
    print("Splitting training dataset into 2 subsets")
    a=dataset.targets[np.array(range(SUBSET_SIZE))]
    unique, counts = np.unique(a, return_counts=True)
    print("The first  one has target counts ", counts)
    b=dataset.targets[np.array(range(SUBSET_SIZE,CIFAR_TRAIN_DATASET_SIZE))]
    unique, counts = np.unique(b, return_counts=True)
    print("The second one has target counts ", counts)

    return dataset1,dataset2

def get_validation_data():
    dataset= datasets.CIFAR10(root='../data', train=False, download=True, transform=CIFAR10Transform.test_transform())
    return dataset
