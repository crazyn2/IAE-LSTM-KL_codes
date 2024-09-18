from __future__ import print_function
import os
import sys
import torch
import lightning as pl
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
from datamodules.wtbi.dsts import WtbiV1
from datamodules.wtbi.dsts import WtbiV2
from utils import seed_worker


class WtbiDmV1(pl.LightningDataModule):

    def __init__(
        self,
        batch_size,
        seed,
        reload=True,
        num_workers=3,
        transform=transforms.Compose([
            transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),
            transforms.ToPILImage(),
            transforms.Pad([3, 3]),
            transforms.ToTensor(),
        ]),
        root_file="data/WTBI/15_avg1_lowPower_data.csv",
        version="v1",
    ):
        """Initialize cifar10 cfg.

        .. seealso::
            See :attr:`~dataset.fmnist` for related property.

        Args:
            batch_size:
                batch_size parameter of dataloader
            time_step:
                frame count to predict frame
            seed:
                dataloader workers's initial seed
            num_pred:
                the predicted frame count

        """
        super().__init__()
        # normal class only one class per training set
        self.save_hyperparameters(ignore='transform')
        pl.seed_everything(seed, workers=True)
        self.batch_size = batch_size
        self.root_file = root_file
        self.num_workers = num_workers
        self.seed = seed
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)

        # transform = transforms.Compose([
        #     transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),
        #     transforms.ToPILImage(),
        #     transforms.Pad([3, 3]),
        #     transforms.ToTensor(),
        # ])
        if version == 'v1':
            self.train_dataset = WtbiV1(
                root_file=self.root_file,
                train=True,
                reload=reload,
                transform=transform,
            )

            # if stage == "test":
            self.test_dataset = WtbiV1(
                root_file=self.root_file,
                train=False,
                reload=reload,
                transform=transform,
            )
        else:
            self.train_dataset = WtbiV2(
                root_file=self.root_file,
                train=True,
                reload=reload,
                transform=transform,
            )

            # if stage == "test":
            self.test_dataset = WtbiV2(
                root_file=self.root_file,
                train=False,
                reload=reload,
                transform=transform,
            )

    def train_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=seed_worker,
                          generator=gen,
                          persistent_workers=True,
                          shuffle=True,
                          drop_last=True)

    def test_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=True,
            generator=gen,
            #   shuffle=True,
            drop_last=True)

    def val_dataloader(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=True,
            generator=gen,
            #   shuffle=True,
            drop_last=True)


if __name__ == '__main__':
    datamodule = WtbiDmV1(batch_size=16, seed=2002)
    for imgs, labels in datamodule.test_dataloader():
        print(imgs.shape, labels)
        break
