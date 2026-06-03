# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import sklearn.model_selection
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from h5py import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.linalg import sqrtm

# from models.Network import config
import scipy.io as sio
# import config

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

import scipy.io as sio
import matplotlib.pyplot as plt
import random

DATASETS_CONFIG = {
    'Houston13': {
        'img_HSI': 'Houston13_HSI.mat',
        'img_LiDAR': 'Houston13_LiDAR.mat',
        'gt': 'Houston13_7gt.mat',
    },
    'Houston18': {
        'img_HSI': 'Houston18_HSI.mat',
        'img_LiDAR': 'Houston18_LiDAR.mat',
        'gt': 'Houston18_7gt.mat',
    },
    'Trento': {
        'img_HSI': 'Trento_HSI.mat',
        'img_LiDAR': 'Trento_LiDAR.mat',
        'gt': 'Trento_3gt.mat',
    },
    'Muufl': {
        'img_HSI': 'Muufl_HSI.mat',
        'img_LiDAR': 'Muufl_LiDAR.mat',
        'gt': 'Muufl_3gt.mat',
    },
    'Augsburg': {
        'img_HSI': 'Augsburg_HSI.mat',
        'img_LiDAR': 'Augsburg_SAR.mat',
        'gt': 'Augsburg_7gt.mat',
    },
    'Berlin': {
        'img_HSI': 'Berlin_HSI.mat',
        'img_LiDAR': 'Berlin_SAR.mat',
        'gt': 'Berlin_7gt.mat',
    },
}


# try:
#     from custom_datasets import CUSTOM_DATASETS_CONFIG
#     DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
# except ImportError:
#     pass

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def pca_process(data, NC):
    [height, width, bands] = data.shape
    temp = np.reshape(data, (height * width, bands))
    pca = PCA(n_components=NC, copy=True, whiten=False)
    temp = pca.fit_transform(temp)
    temp = np.reshape(temp, (height, width, NC))
    return temp


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG, CUSTOM_DATASETS_CONFIG=None):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    # palette = None

    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder  # + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', False):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                              desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                reporthook=t.update_to)
    elif not os.path.isdir(folder):
        print("WARNING: {} is not downloadable.".format(dataset_name))

    # 读取数据集
    if dataset_name == 'Houston13':
        # Load the image
        img_HSI = np.asarray(sio.loadmat(folder + 'HSI_data.mat')['HSI_data']).astype(np.float32)
        img_LiDAR = np.asarray(sio.loadmat(folder + 'LiDAR_data.mat')['LiDAR_data']).astype(np.float32)
        # img_HSI = np.asarray(open_file(folder + 'Houston13_LiDAR.mat')['LiDAR_data']).astype(np.float32)
        # img_LiDAR = np.asarray(open_file(folder + 'Houston13_LiDAR.mat')['LiDAR_data']).astype(np.float32)
        # img_LiDAR = np.asarray(open_file(folder + 'Houston13_HSI.mat')['HSI_data']).transpose(1, 2, 0).astype(np.float32)

        # rgb_bands = [13, 20, 33]

        gt = np.asarray(sio.loadmat(folder + 'All_Label.mat')['All_Label'])
        ignored_labels = [0]

    elif dataset_name == 'Houston18':
        # Load the image
        img_HSI = np.asarray(sio.loadmat(folder + 'Houston18_HSI.mat')['HSI_data']).transpose(1, 2, 0).astype(np.float32)
        img_LiDAR = np.asarray(sio.loadmat(folder + 'Houston18_LiDAR.mat')['LiDAR_data']).astype(np.float32)
        # img_HSI = np.asarray(open_file(folder + 'Houston18_LiDAR.mat')['LiDAR_data']).astype(np.float32)
        # img_LiDAR = np.asarray(open_file(folder + 'Houston18_LiDAR.mat')['LiDAR_data']).astype(np.float32)
        # img_LiDAR = np.asarray(open_file(folder + 'Houston18_HSI.mat')['HSI_data']).transpose(1, 2, 0).astype(np.float32)

        # rgb_bands = [13, 20, 33]

        gt = np.asarray(sio.loadmat(folder + 'Houston18_7gt.mat')['All_Label'])
        ignored_labels = [0]

    elif dataset_name == 'Trento':
        # Load the image
        img_HSI = np.asarray(sio.loadmat(folder + 'HSI_data.mat')['HSI_data']).transpose(1, 2, 0).astype(np.float32)
        img_LiDAR = np.asarray(sio.loadmat(folder + 'LiDAR_data.mat')['LiDAR_data']).astype(np.float32)

        # rgb_bands = [20, 30, 30]

        gt = np.asarray(sio.loadmat(folder + 'All_Label.mat')['All_Label'])
        ignored_labels = [0]


    elif dataset_name == 'Muufl':
        # Load the image
        img_HSI = np.asarray(sio.loadmat(folder + 'Muufl_HSI.mat')['HSI_data']).transpose(1, 2, 0).astype(np.float32)
        img_LiDAR = np.asarray(sio.loadmat(folder + 'Muufl_LiDAR.mat')['LiDAR_data']).astype(np.float32)

        # rgb_bands = [20, 30, 30]

        gt = np.asarray(sio.loadmat(folder + 'Muufl_3gt.mat')['All_Label'])
        ignored_labels = [0]


    elif dataset_name == 'Augsburg':
        # Load the image
        img_HSI = np.asarray(sio.loadmat(folder + 'Augsburg_HSI.mat')['HSI_data']).transpose(1, 2, 0).astype(np.float32)
        img_LiDAR = np.asarray(sio.loadmat(folder + 'Augsburg_SAR.mat')['SAR_data']).transpose(1, 2, 0).astype(np.float32)

        # rgb_bands = [20, 30, 30]

        gt = np.asarray(sio.loadmat(folder + 'Augsburg_7gt.mat')['All_Label'])
        ignored_labels = [0]


    elif dataset_name == 'Berlin':
        # Load the image
        img_HSI = np.asarray(sio.loadmat(folder + 'Berlin_HSI.mat')['HSI_data']).transpose(1, 2, 0).astype(np.float32)
        img_LiDAR = np.asarray(sio.loadmat(folder + 'Berlin_SAR.mat')['SAR_data']).transpose(1, 2, 0).astype(np.float32)

        # rgb_bands = [20, 30, 30]

        gt = np.asarray(sio.loadmat(folder + 'Berlin_7gt.mat')['All_Label'])
        ignored_labels = [0]

    else:
        # Custom dataset
        img_HSI, img_LiDAR, gt= CUSTOM_DATASETS_CONFIG[dataset_name][
            'loader'](folder)
        # img_LiDAR, gt, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name][
        #     'loader'](folder)


    nan_mask = np.isnan(img_HSI.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img_HSI[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))
    # Normalization
    img_HSI = np.asarray(img_HSI, dtype='float32')

    nan_mask = np.isnan(img_LiDAR.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img_LiDAR[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # 归一化，但是什么归一化不知道，随机选择一种也行
    m1, n1, d1 = img_HSI.shape[0], img_HSI.shape[1], img_HSI.shape[2]
    img_HSI= img_HSI.reshape((m1*n1,-1))
    img_HSI = img_HSI/img_HSI.max()
    img_HSI_temp = np.sqrt(np.asarray((img_HSI**2).sum(1)))
    img_HSI_temp = np.expand_dims(img_HSI_temp,axis=1)
    img_HSI_temp = img_HSI_temp.repeat(d1,axis=1)
    img_HSI_temp[img_HSI_temp==0]=1
    img_HSI = img_HSI/img_HSI_temp
    img_HSI_ = np.reshape(img_HSI,(m1,n1,-1))
    img_LiDAR_ = img_LiDAR.reshape(img_LiDAR.shape[0], img_LiDAR.shape[1], -1)
    img_HSI_Norm = np.zeros_like(img_HSI_)
    img_LiDAR_Norm = np.zeros_like(img_LiDAR_)




    scaler = StandardScaler()
    for i in range(img_HSI_.shape[2]):
        img_HSI_Norm[:, :, i] = scaler.fit_transform(img_HSI_[:, :, i])
    for i in range(img_LiDAR_Norm.shape[2]):
        img_LiDAR_Norm[:, :, i] = scaler.fit_transform(img_LiDAR_[:, :, i])
        # img_LiDAR_Norm[:, :, i] = (img_LiDAR_Norm[:, :, i] - np.min(img_LiDAR_Norm[:, :, i])) / (np.max(img_LiDAR_Norm[:, :, i]) - np.min(img_LiDAR_Norm[:, :, i]))

    data_tuple = (img_HSI_Norm, img_LiDAR_Norm)
    return data_tuple, gt, ignored_labels


# 取patch
def build_patch(Data, gt, patchsize, samples_type, train_size, num_samples_per_class=40):
    data_pad = []
    pad_width = np.int32(np.floor(patchsize / 2))

    for data in Data:
        bands = data.shape[-1]
        data_pad_ = np.empty((data.shape[0] + 2 * pad_width, data.shape[1] + 2 * pad_width, bands), dtype='float32')
        for i in range(bands):
            temp = data[:, :, i]
            data_pad_[:, :, i] = np.pad(temp, pad_width, 'symmetric')
        data_pad.append(data_pad_)

    indices = np.nonzero(gt)
    X = list(zip(*indices))
    y = gt[indices].ravel()
    unique_classes = np.unique(y)

    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)

    if samples_type == 'ratio':
        train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=23)
    elif samples_type == 'num':
        train_indices, test_indices = [], []
        for cls in unique_classes:
            cls_indices = [i for i, val in enumerate(y) if val == cls]
            np.random.shuffle(cls_indices)
            cls_train = cls_indices[:num_samples_per_class]
            cls_test = cls_indices
            train_indices.extend([X[i] for i in cls_train])
            test_indices.extend([X[i] for i in cls_test])
    else:
        raise ValueError("samples_type must be 'ratio' or 'num'")

    train_idx_t = tuple(zip(*train_indices))
    test_idx_t = tuple(zip(*test_indices))
    train_gt[train_idx_t] = gt[train_idx_t]
    test_gt[test_idx_t] = gt[test_idx_t]

    def extract_patches(data_pad, index_set):
        N = len(index_set[0])
        result = []
        for i, data in enumerate(data_pad):
            bands = data.shape[-1]
            patches = np.empty((N, bands, patchsize, patchsize), dtype='float32')
            for j in range(N):
                x, y = index_set[0][j] + pad_width, index_set[1][j] + pad_width
                patch = data[x - pad_width:x + pad_width + 1, y - pad_width:y + pad_width + 1, :]
                patch = patch.transpose(2, 0, 1)  # to (C, H, W)
                patches[j] = patch
            result.append(patches)
        return result

    train_index = np.where(train_gt != 0)
    test_index = np.where(test_gt != 0)
    data_patch_train = extract_patches(data_pad, train_index)
    data_patch_test = extract_patches(data_pad, test_index)

    return (data_patch_train, train_gt), (data_patch_test, test_gt)

class MyDataset(Dataset):
    def __init__(self, data_patch, label_map):
        super(MyDataset2, self).__init__()
        self.data_patch = data_patch
        self.height, self.width = label_map.shape

        self.indices_2d = np.array(np.nonzero(label_map)).T
        self.labels = label_map[self.indices_2d[:, 0], self.indices_2d[:, 1]]

        self.indices_flat = self.indices_2d[:, 0] * self.width + self.indices_2d[:, 1]

    def __getitem__(self, idx):
        x1 = torch.from_numpy(self.data_patch[0][idx])
        x2 = torch.from_numpy(self.data_patch[1][idx])
        label = torch.tensor(self.labels[idx] - 1).to(torch.long)
        index = int(self.indices_flat[idx])
        return (x1, x2), label, index

    def __len__(self):
        return len(self.labels)

def load_data(dataset_name, data_path, patchsize, samples_type, train_size, num_samples_per_class):
    data_tuple, gt, ignored_labels = get_dataset(dataset_name, data_path)

    (data_patch_train, train_gt), (data_patch_test, test_gt) = build_patch(
        Data=data_tuple,
        gt=gt,
        patchsize=patchsize,
        samples_type=samples_type,
        train_size=train_size,
        num_samples_per_class=num_samples_per_class
    )

    train_dataset = MyDataset(data_patch=data_patch_train, label_map=train_gt)
    test_dataset = MyDataset(data_patch=data_patch_test, label_map=test_gt)

    return train_dataset, test_dataset, ignored_labels

