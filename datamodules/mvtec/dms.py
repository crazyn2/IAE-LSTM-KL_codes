from __future__ import print_function
import os
import sys
import torch
import lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
from datamodules.mvtec.dsts import MvTecV1
from utils import seed_worker


class MvTecDmV1(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 seed,
                 normal_class=1,
                 num_workers=3,
                 root="data/mvtec",
                 transform=transforms.Compose([
                     transforms.Resize([256, 256]),
                     transforms.ToTensor(),
                 ]),
                 reload=True):
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
        self.root = root
        self.num_workers = num_workers
        self.seed = seed
        self.object_name = normal_class
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)

        self.train_dataset = MvTecV1(root=self.root,
                                     object_name=self.object_name,
                                     train=True,
                                     transform=transform,
                                     reload=reload)

        # if stage == "test":
        self.test_dataset = MvTecV1(root=self.root,
                                    object_name=self.object_name,
                                    train=False,
                                    transform=transform,
                                    reload=reload)

    def train_dataloader(self):
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=seed_worker,
                          generator=self.gen,
                          persistent_workers=True,
                          shuffle=True,
                          drop_last=True)

    def test_dataloader(self):
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=True,
            generator=self.gen,
            #   shuffle=True,
            drop_last=True)

    def val_dataloader(self):
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=True,
            generator=self.gen,
            #   shuffle=True,
            drop_last=True)


if __name__ == '__main__':
    datamodule = MvTecDmV1(
        batch_size=16,
        seed=2002,
    )
    for imgs, mask, labels in datamodule.train_dataloader():
        print(imgs.shape, labels)
        # break
