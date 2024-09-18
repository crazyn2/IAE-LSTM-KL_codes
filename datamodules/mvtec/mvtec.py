from __future__ import print_function
import os
import sys
import torch
from glob import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import lightning as pl
from random import sample
import random
from PIL import Image
from torch.utils.data import Dataset
from imageio.v2 import imread

from torchvision.datasets import CIFAR10
# from .preprocessing import global_contrast_normalization,get_target_label_idx
import numpy as np

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
# from utils import gray2rgb
from utils import resize
# from utils import generate_coords_svdd
from utils import crop_image_CHW
from preprocessing import global_contrast_normalization, get_target_label_idx


def generate_coords(H, W, K):
    h = np.random.randint(0, H - K + 1)
    w = np.random.randint(0, W - K + 1)
    return h, w


def generate_coords_position(H, W, K):
    # p1
    p1 = generate_coords(H, W, K)
    h1, w1 = p1

    pos = np.random.randint(8)
    # p2
    J = K // 4

    K3_4 = 3 * K // 4
    h_dir, w_dir = pos_to_diff[pos]
    h_del, w_del = np.random.randint(J, size=2)

    h_diff = h_dir * (h_del + K3_4)
    w_diff = w_dir * (w_del + K3_4)

    h2 = h1 + h_diff
    w2 = w1 + w_diff

    h2 = np.clip(h2, 0, H - K)
    w2 = np.clip(w2, 0, W - K)

    p2 = (h2, w2)

    return p1, p2, pos


def generate_coords_svdd(H, W, K):
    # p1
    p1 = generate_coords(H, W, K)
    h1, w1 = p1

    # p2
    J = K // 32

    h_jit, w_jit = 0, 0

    while h_jit == 0 and w_jit == 0:
        h_jit = np.random.randint(-J, J + 1)
        w_jit = np.random.randint(-J, J + 1)

    h2 = h1 + h_jit
    w2 = w1 + w_jit

    h2 = np.clip(h2, 0, H - K)
    w2 = np.clip(w2, 0, W - K)

    p2 = (h2, w2)

    return p1, p2


pos_to_diff = {
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, -1),
    4: (0, 1),
    5: (1, -1),
    6: (1, 0),
    7: (1, 1)
}


class MVTecDataset(Dataset):
    def __init__(self,
                 data_dir="./data/mvtec",
                 object="bottle",
                 mode='train',
                 transform=None,
                 mask_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.object = object
        fpattern = os.path.join(self.data_dir, f'{self.object}/{mode}/*/*.png')
        fpaths = sorted(glob(fpattern))
        if mode == 'test':
            fpaths_abn = list(
                filter(
                    lambda fpath: os.path.basename(os.path.dirname(fpath)) !=
                    'good', fpaths))

            fpaths_nor = list(
                filter(
                    lambda fpath: os.path.basename(os.path.dirname(fpath)) ==
                    'good', fpaths))

            self.image_paths = np.concatenate([fpaths_nor, fpaths_abn])
            # Mask
            fmask_pattern = os.path.join(
                self.data_dir, f'{self.object}/ground_truth/*/*.png')
            self.fmask_paths = sorted(glob(fmask_pattern))
            masks = np.asarray(list(map(Image.open, fpaths)))
            num_abn = masks.shape[0]
            num_nor = len(fpaths_nor)

            masks[masks <= 128] = 0
            masks[masks > 128] = 255
            results = np.zeros((num_abn + num_nor, ) + masks.shape[1:],
                               dtype=masks.dtype)
            results[num_nor:] = masks
            self.masks = results

            self.labels = [0] * len(fpaths_nor) + [1] * len(fpaths_abn)

        else:
            self.image_paths = fpaths
            self.labels = [0] * len(self.images)
            self.masks = np.zeros()
        # 获取所有图像文件路径
        self.image_paths = [
            os.path.join(data_dir, img) for img in os.listdir(data_dir)
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask = self.masks[idx]
        label = self.labels[idx]
        mask = self.masks[idx]
        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, label


class SVDD_Dataset(Dataset):
    def __init__(self, root_path, obj, mode, K=64):
        super().__init__()

        fpattern = os.path.join(root_path, f'{obj}/{mode}/*/*.png')
        fpaths = sorted(glob(fpattern))

        if mode == 'test':
            fpaths1 = list(
                filter(
                    lambda fpath: os.path.basename(os.path.dirname(fpath)) !=
                    'good', fpaths))
            fpaths2 = list(
                filter(
                    lambda fpath: os.path.basename(os.path.dirname(fpath)) ==
                    'good', fpaths))

            images1 = np.asarray(list(map(imread, fpaths1)))
            images2 = np.asarray(list(map(imread, fpaths2)))
            images = np.concatenate([images1, images2])

        else:
            images = np.asarray(list(map(imread, fpaths)))

        if images.shape[-1] != 3:
            images = gray2rgb(images)
        images = list(map(resize, images))
        self.arr = np.asarray(images)
        self.K = K

    def __len__(self):
        N = self.arr.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):

        p1, p2 = generate_coords_svdd(256, 256, self.K)

        image = self.arr[idx]

        patch1 = crop_image_CHW(image, p1, self.K)
        patch2 = crop_image_CHW(image, p2, self.K)

        return patch1, patch2


class MVTECDataModel(pl.LightningDataModule):
    def __init__(self,
                 batch_size,
                 normal_class,
                 radio=0,
                 num_workers=8,
                 root="./data/mvtec",
                 dataset_name="mvtec",
                 object_name="bottle"):
        super().__init__()
        # normal class only one class per training set
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.object_name = object_name
        self.root = root
        # 污染数据比例
        self.radio = radio
        self.normal_class = normal_class
        self.num_workers = num_workers
        # self.normal_classes = tuple([normal_class])
        # self.outlier_classes = list(range(0, 10))
        # self.outlier_classes.remove(normal_class)

    def setup(self, stage: str) -> None:

        if stage == 'test':
            fpattern = os.path.join(self.root,
                                    f'{self.object_name}/{stage}/*/*.png')
            fpaths = sorted(glob(fpattern))
            fpaths_abn = list(
                filter(
                    lambda fpath: os.path.basename(os.path.dirname(fpath)) !=
                    'good', fpaths))
            fpaths_nor = list(
                filter(
                    lambda fpath: os.path.basename(os.path.dirname(fpath)) ==
                    'good', fpaths))

            images1 = np.asarray(list(map(Image.open, fpaths_abn)))
            images2 = np.asarray(list(map(Image.open, fpaths_nor)))
            images = np.concatenate([images1, images2])

        else:
            # stage == "fit"
            images = np.asarray(list(map(Image.open, fpaths)))

        if images.shape[-1] != 3:
            images = gray2rgb(images)
        images = list(map(resize, images))
        images = np.asarray(images)
        # Pre-computed min and max values (after applying GCN) from train data per class
        # global_contrast_normalization

    def train_dataloader(self):
        return DataLoader(self.train_cifar10,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_cifar10,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_cifar10,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)


def get_gcn():
    os.makedirs("log/mvtec", exist_ok=True)

    print(os.path.dirname(__file__))
    print(os.path.dirname(os.path.dirname(__file__)))
    # i = 0
    # for inputs, labels in data_loader_train:
    #     print(inputs.shape)
    #     # plot_images_grid(inputs, export_img="log/cifar10/train_%d" % i)
    #     break
    #     i += 1
    train_set_full = CIFAR10(
        root="./data/",
        train=True,
        #  download=True,
        transform=None,
        target_transform=None)

    MIN = []
    MAX = []
    for normal_classes in range(10):
        train_idx_normal = get_target_label_idx(train_set_full.targets,
                                                normal_classes)
        train_set = Subset(train_set_full, train_idx_normal)

        _min_ = []
        _max_ = []
        for idx in train_set.indices:
            # print(train_set.dataset.data[idx])
            gcm = global_contrast_normalization(
                torch.from_numpy(train_set.dataset.data[idx]).double(), 'l1')
            _min_.append(gcm.min())
            _max_.append(gcm.max())
        MIN.append(np.min(_min_))
        MAX.append(np.max(_max_))
    print(list(zip(MIN, MAX)))


if __name__ == '__main__':
    # cifar10 = MVTECDataModel(batch_size=100, normal_class=1, radio=0.1)
    MVTecDataset(mode="test")
    # cifar10.setup("fit")
    # cifar10.setup("test")
