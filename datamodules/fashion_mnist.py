from torch.utils.data import Subset
from torchvision.datasets import FashionMNIST
import torch
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
import os
import sys
import lightning as pl
from random import sample
import random

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
from preprocessing import global_contrast_normalization


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fmnist_dataset(normal_class: list, batch_size, od=False):
    """Loads the dataset."""
    transform = transforms.Compose([
        transforms.Pad((2, 2)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    target_transform = transforms.Lambda(lambda x: int(x not in normal_class))
    train_fmnist = FashionMNIST(root="./data/",
                                train=True,
                                transform=transform,
                                target_transform=target_transform,
                                download=True)
    test_fmnist = FashionMNIST(root="./data/",
                               train=False,
                               transform=transform,
                               target_transform=target_transform,
                               download=True)
    # print(train_fmnist.targets)
    train_indices = [
        idx for idx, target in enumerate(train_fmnist.targets)
        if target in normal_class
    ]
    data_loader_train = DataLoader(Subset(train_fmnist, train_indices),
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True)
    data_loader_test = DataLoader(test_fmnist,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    if od:
        data_loader_train = DataLoader(train_fmnist,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True)
    return data_loader_train, data_loader_test


class FMNISTDm(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 normal_class,
                 seed,
                 radio=0,
                 num_workers=4,
                 root="./data/",
                 padding=False,
                 gcn=True,
                 transform=None,
                 loadingv1=False):
        super().__init__()
        self.save_hyperparameters(ignore='transform')
        self.batch_size = batch_size
        self.padding = padding
        # self.prepare_data_per_node = Falseself.center = self.center.to(self.device)
        self.root = root
        # 污染数据比例
        self.radio = radio
        self.normal_class = normal_class
        self.num_workers = num_workers
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.seed = seed
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)
        self.gcn = gcn
        self.loadingv1 = loadingv1

        # def setup(self, stage: str) -> None:

        # Pre-computed min and max values (after applying GCN) from train data per class
        # global_contrast_normalization
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]
        gcn_transform = [
            transforms.Lambda(
                lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([min_max[self.normal_class][0]], [
                min_max[self.normal_class][1] - min_max[self.normal_class][0]
            ])
        ]
        transformers_list = []
        if self.padding:
            transformers_list.append(transforms.Pad((2, 2)))

        transformers_list.append(transforms.ToTensor())

        if self.gcn:
            transformers_list += gcn_transform

        else:
            transformers_list.append(transforms.Normalize([0.5], [0.5]))
        if transform is None:
            transform = transforms.Compose(transformers_list)
        # if self.p
        target_transform = transforms.Lambda(
            lambda x: int(x in self.outlier_classes))

        # if stage == "fit":
        train_fmnist = FashionMNIST(root=self.root,
                                    train=True,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=False)

        train_indices = [
            idx for idx, target in enumerate(train_fmnist.targets)
            if target in self.normal_classes
        ]
        dirty_indices = [
            idx for idx, target in enumerate(train_fmnist.targets)
            if target not in self.normal_classes
        ]
        self.radio / (1 - self.radio)
        train_indices += sample(
            dirty_indices,
            int(len(train_indices) * self.radio / (1 - self.radio)))
        # dataloader shuffle=True will mix the order of normal and abnormal
        # extract the normal class of fmnist train dataset
        self.train_fmnist = Subset(train_fmnist, train_indices)

        # if stage == "test":
        self.test_fmnist = FashionMNIST(root=self.root,
                                        train=False,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=False)

    def train_dataloader(self):
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)
        return DataLoader(self.train_fmnist,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=seed_worker,
                          persistent_workers=True,
                          generator=self.gen,
                          shuffle=True,
                          drop_last=True)

    def test_dataloader(self):
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)
        return DataLoader(self.test_fmnist,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          worker_init_fn=seed_worker,
                          generator=self.gen,
                          drop_last=True)

    def val_dataloader(self):
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)
        return DataLoader(self.test_fmnist,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          worker_init_fn=seed_worker,
                          generator=self.gen,
                          drop_last=True)


if __name__ == "__main__":
    cifar10 = FMNISTDm(batch_size=64,
                       normal_class=1,
                       radio=0.1,
                       gcn=False,
                       seed=0)
    # cifar10 = CIFAR10GmTranV1(batch_size=100,
    #                           normal_class=1,
    #                           radio=0.1,
    #                           seed=0,
    #                           gcn=False)
    cifar10.setup("fit")
    train_data = cifar10.train_dataloader()
    print(len(train_data))
    for inputs, labels in train_data:
        print(inputs.shape)
        for i, x in enumerate(inputs):
            print(x.shape)
            break
        break
