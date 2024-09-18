import glob
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class MvTecV1(Dataset):

    def __init__(
        self,
        object_name,
        root: str = "data/mvtec",
        cache_root: str = "data/mvtec_cache",
        train=True,
        transform=None,
        mask_transform=None,
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        reload=True,
    ):
        super().__init__()
        self.transform = transform
        self.mask_transform = mask_transform
        self.root = root
        self.train = train
        contents = os.listdir(self.root)
        classes = sorted([
            content for content in contents
            if os.path.isdir(os.path.join(self.root, content))
        ])
        object_name = classes[object_name]
        self.cache_root = os.path.join(cache_root, object_name)
        self.train_imgs_cache = os.path.join(self.cache_root, "train_imgs.npy")
        self.test_imgs_cache = os.path.join(self.cache_root, "test_imgs.npy")
        self.masks_cache = os.path.join(self.cache_root, "masks.npy")
        self.labels_cache = os.path.join(self.cache_root, "labels.npy")
        os.makedirs(self.cache_root, exist_ok=True)
        if self.train:
            if reload:
                self.imgs = np.load(self.train_imgs_cache)
            else:
                self.mvtec_paths = sorted(
                    glob.glob(
                        os.path.join(self.root, object_name, "train/*/*.png")))
                self.imgs = np.asarray(list(map(Image.open, self.mvtec_paths)))
                np.save(self.train_imgs_cache, self.imgs)
        else:
            if reload:
                self.imgs = np.load(self.test_imgs_cache)
                self.masks = np.load(self.masks_cache)
                self.labels = np.load(self.labels_cache)
            else:
                mvtec_paths = glob.glob(
                    os.path.join(self.root, object_name, "test/*/*.png"))
                normal_paths = sorted(
                    list(filter(lambda fpath: 'good' in fpath, mvtec_paths)))
                abnormal_paths = sorted(
                    list(filter(lambda fpath: 'good' not in fpath,
                                mvtec_paths)))
                self.mvtec_paths = normal_paths + abnormal_paths
                self.imgs = np.asarray(list(map(Image.open, self.mvtec_paths)))
                mask_paths = sorted(
                    glob.glob(
                        os.path.join(self.root, object_name,
                                     "ground_truth/*/*.png")))
                abn_masks = np.asarray(list(map(Image.open, mask_paths)))
                self.masks = np.zeros(
                    (len(normal_paths) + abn_masks.shape[0], ) +
                    abn_masks.shape[1:],
                    dtype=abn_masks.dtype)
                self.masks[len(normal_paths):] = abn_masks
                self.labels = [0] * len(normal_paths) + [1] * len(
                    abnormal_paths)
                np.save(self.test_imgs_cache, self.imgs)
                np.save(self.masks_cache, self.masks)
                np.save(self.labels_cache, self.labels)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        # image = Image.open(self.mvtec_paths[index]).convert("RGB")
        # print(np.array(image))
        # print('*' * 20)
        # print(self.mvtec_imgs[index])
        image = Image.fromarray(self.imgs[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.train:
            # return image, np.zeros(image.shape), 0
            return image, np.zeros(image.shape), 0
        else:
            mask = self.masks[index]
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            return image, mask, self.labels[index]


if __name__ == "__main__":
    # for i in range(15):
    #     dataset = MvTecV1(train=True, object_name=i, reload=False)
    #     dataset = MvTecV1(train=True, object_name=i, reload=False)
    dataset = MvTecV1(train=False, object_name=4, reload=False)
    # print(dataset.__len__())
    # print(dataset[54][0].size)
