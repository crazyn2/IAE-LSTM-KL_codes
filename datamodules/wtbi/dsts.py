from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


class WtbiV1(Dataset):

    def __init__(self,
                 root: str = "data/WTBI",
                 root_file: str = "data/WTBI/15_avg1_lowPower_data.csv",
                 train: bool = True,
                 radio=None,
                 reload=False,
                 transform=None) -> None:
        self.root = root
        self.root_file = root_file
        self.transform = transform
        self.train = train
        self.radio = radio
        self.data = pd.read_csv(self.root_file)
        self.data.drop(columns=['time', 'group'], inplace=True)
        # print(self.data[self.data['frozen'] == 0].shape)
        self.data = self.data.apply(pd.to_numeric)
        self.data = self.data.apply(lambda x: (x - np.min(x)) /
                                    (np.max(x) - np.min(x)))
        if self.radio is not None:
            self.train_cache = os.path.join(self.root, "train_data_radio.npy")
            self.test_cache = os.path.join(self.root, "test_data_radio.npy")
            self.labels_cache = os.path.join(self.root,
                                             "test_labels_radio.npy")
        else:
            self.train_cache = os.path.join(self.root, "train_data.npy")
            self.test_cache = os.path.join(self.root, "test_data.npy")
            self.labels_cache = os.path.join(self.root, "test_labels.npy")
        if reload:
            self.train_data = np.load(self.train_cache)
            self.test_data = np.load(self.test_cache)
            self.test_labels = np.load(self.labels_cache)
        else:
            self.setup()

    def setup(self):
        normal = self.data[self.data['frozen'] == 0].drop(
            columns=['frozen']).reset_index(drop=True)
        abnormal = self.data[self.data['frozen'] == 1].drop(
            columns=['frozen']).reset_index(drop=True)
        normal_img = self.split(normal)
        abnormal_img = self.split(abnormal)
        if self.radio is not None:
            # normal_indices = np.random.choice(range(normal_img.shape[0]),
            #                                   size=abnormal_img.shape[0],
            #                                   replace=False)
            # radio_normal = normal_img[normal_indices]
            # abnormal_indices = np.random.choice(range(abnormal_img.shape[0]),
            #                                     size=abnormal_img.shape[0] / 2,
            #                                     replace=False)
            left_normal, radio_normal = train_test_split(
                normal_img, test_size=abnormal_img.shape[0], random_state=2042)
            radio_abnormal, test_radio_abnormal = train_test_split(
                abnormal_img,
                test_size=int(abnormal_img.shape[0] / 2),
                random_state=2042)
            # 异常样本数量
            abnormal_count = int(radio_normal.shape[0] * self.radio /
                                 (1 - self.radio))
            _, train_abnormal = train_test_split(radio_abnormal,
                                                 test_size=abnormal_count,
                                                 random_state=2042)
            self.train_data = np.vstack((radio_normal, train_abnormal))
            labels_0 = np.zeros(left_normal.shape[0])
            labels_1 = np.ones(test_radio_abnormal.shape[0])
            self.test_data = np.vstack((left_normal, test_radio_abnormal))
            self.test_labels = np.concatenate((labels_0, labels_1))
        #
        else:
            self.train_data, test_normal = train_test_split(normal_img,
                                                            test_size=0.3,
                                                            random_state=2042)
            labels_0 = np.zeros(test_normal.shape[0])
            labels_1 = np.ones(abnormal_img.shape[0])
            self.test_data = np.vstack((test_normal, abnormal_img))
            for i in range(self.test_data.shape[0]):
                plt.imsave('data/WTBI/test/%05d.png' % i,
                           self.test_data[i],
                           cmap='gray')
            self.test_labels = np.concatenate((labels_0, labels_1))
        np.save(self.train_cache, self.train_data)
        np.save(self.test_cache, self.test_data)
        np.save(self.labels_cache, self.test_labels)

    def split(self, df):
        # print(df.shape)
        # group_key = (df.index // df.shape[-1]) + 1

        # grouped_df = df.groupby(group_key)
        # grouped_arrays = grouped_df.apply(lambda x: x.to_numpy()).tolist()
        grouped_dfs = [
            df.iloc[i:i + df.shape[-1]]
            for i in range(0, df.shape[0] - df.shape[-1])
        ]
        # print(grouped_dfs[-1].shape)
        return np.hstack((grouped_dfs[:-1], ))
        # print(len(grouped_arrays))
        # return np.hstack((grouped_arrays[:-1], ))

    def __getitem__(self, index):
        if self.train:
            data = self.train_data[index]
            label = 0
        else:
            data = self.test_data[index]
            label = self.test_labels[index]
        if self.transform:
            data = self.transform(data)

        return data, label

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class WtbiV2(WtbiV1):

    def __init__(self,
                 root: str = "data/WTBI",
                 root_file: str = "data/WTBI/15_avg1_lowPower_data.csv",
                 train: bool = True,
                 reload=False,
                 transform=None) -> None:
        self.root = root
        self.root_file = root_file
        self.transform = transform
        self.train = train
        self.data = pd.read_csv(self.root_file)
        self.data.drop(columns=['time', 'group'], inplace=True)
        self.data = self.data.apply(pd.to_numeric)
        self.data = self.data.apply(lambda x: (x - np.min(x)) /
                                    (np.max(x) - np.min(x)))
        self.train_cache = os.path.join(self.root, "train_datav2.npy")
        self.test_cache = os.path.join(self.root, "test_datav2.npy")
        self.labels_cache = os.path.join(self.root, "test_labelsv2.npy")
        if reload:
            self.train_data = np.load(self.train_cache)
            self.test_data = np.load(self.test_cache)
            self.test_labels = np.load(self.labels_cache)
        else:
            self.setup()

    def split(self, df):
        # print(df.shape)
        group_key = (df.index // df.shape[-1]) + 1

        grouped_df = df.groupby(group_key)
        grouped_arrays = grouped_df.apply(lambda x: x.to_numpy()).tolist()
        # grouped_dfs = [
        #     df.iloc[i:i + df.shape[-1]]
        #     for i in range(0, df.shape[0] - df.shape[-1])
        # ]
        # print(grouped_dfs[-1].shape)
        # return np.hstack((grouped_dfs[:-1], ))
        # print(len(grouped_arrays))
        return np.hstack((grouped_arrays[:-1], ))


if __name__ == "__main__":
    dataset = WtbiV1(train=True, reload=False, radio=0.05)
    print(dataset.__len__())
    # dataset = WtbiV1(train=True, reload=True)
    # print(dataset.__len__())
    # print(dataset[54][0].shape)
