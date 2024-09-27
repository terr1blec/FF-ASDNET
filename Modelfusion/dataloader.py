import os
import pandas as pd
import random

import numpy as np
import numba
from numba import jit
import gc
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data
from torchvision import transforms
import pathlib


class pallmediaMyDataset(data.Dataset):
    def __init__(
        self,
        args,
        dvroot_dir,
        zsroot_dir,
        lroot_dir,
        islongimg,
        root_sample,
        roun,
        train,
        padding=0,
        weights=None,
    ):
        super().__init__()
        self.size = {
            "media1": 234,
            "media2": 286,
            "media3": 44,
            "media4": 225,
            "media5": 186,
            "media6": 64,
            "media7": 80,
            "media8": 83,
            "media9": 152,
            "media10": 128,
            "media11": 135,
            "media12": 58,
            "media13": 75,
            "media14": 86,
        }

        w = int(720 * 1.2)
        h = int(576 * 1.2)
        self.round = roun
        self.dvroot_dir = dvroot_dir
        self.zsroot_dir = zsroot_dir
        self.lroot_dir = lroot_dir
        self.root_sample = root_sample
        self.train = train
        self.islong = islongimg
        self.padding = padding
        setup_seed(3407)

        fo = open(f"{self.root_sample}/participant14.txt", "r")
        participants = []
        myFile = fo.read()
        myRecords = myFile.split("\n")
        for y in range(0, len(myRecords) - 1):
            participants.append(myRecords[y].split("\t"))  # 把表格数据存入

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([224, 224]),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        if self.padding == 0:
            ltransform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize([50, 2000]),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            ltransform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(25),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        trainset = []

        fo = open(f"{self.root_sample}/14/{self.round}/{train}sample.txt", "r")
        myFile = fo.read()
        myRecords = myFile.split("\n")
        for y in range(0, len(myRecords) - 1):
            trainset.append(myRecords[y].split("\t"))  # 把表格数据存入

        self.samples = trainset

        label = []
        for i in range(len(self.samples)):
            if self.samples[i][1][0] == "n":  # 新数据集是t,旧数据集是n
                label.append(0)
            else:
                label.append(1)
        self.labels = label

        print(train)
        print(f"{len(self.samples)} samples")

        self.transform = transform
        self.ltransform = ltransform

        pos = np.sum(self.labels)
        neg = len(self.labels) - pos
        self.weights = torch.FloatTensor([1, neg / pos])

    def __len__(self):
        return len(self.samples)

    def getsamples(self):
        return self.samples

    def getlabel(self):
        return self.labels

    def __getitem__(self, index):
        dvarray = cv2.imread(
            f"{self.dvroot_dir}/{self.round}/{self.samples[index][0]}/{self.samples[index][1]}.jpg"
        )
        dvsample = self.transform(dvarray)
        zsarray = cv2.imread(
            f"{self.zsroot_dir}/{self.round}/{self.samples[index][0]}/{self.samples[index][1]}.jpg"
        )
        zssample = self.transform(zsarray)
        if self.islong == True:
            larray = cv2.imread(
                f"{self.lroot_dir}/{self.samples[index][0]}/{self.samples[index][1]}.jpg"
            )
            lsample = self.ltransform(larray)
        else:
            lsample = []
        # return sample,self.labels[index],self.size[self.samples[index][0]]
        # sample=self.images[index]
        label = self.labels[index]
        bz = self.size[self.samples[index][0]]

        return dvsample, zssample, lsample, label, bz, self.padding


def _collate_fn(batch):
    def func(p):
        return p[2].size(2)

    if batch[0][5] == 0:
        minibatch_size = len(batch)
        inputs = []
        lengths = torch.IntTensor(minibatch_size)
        targets = []
        input_size_orig = []
        phoneids = []
        dvsample = torch.zeros(
            minibatch_size, 3, batch[0][0].size(1), batch[0][0].size(2)
        )
        zssample = torch.zeros(
            minibatch_size, 3, batch[0][1].size(1), batch[0][1].size(2)
        )
        for x in range(minibatch_size):
            sample = batch[x]
            target = sample[3]
            targets.append(target)
            input_size_orig.append(sample[4])
            dvsample[x] = batch[x][0]
            zssample[x] = batch[x][1]

        label = torch.tensor(targets, dtype=torch.long)
        return dvsample, zssample, inputs, label, np.array(input_size_orig)
    else:
        longest_sample = max(batch, key=func)[2]
        minibatch_size = len(batch)
        max_seqlength = longest_sample.size(2)
        inputs = torch.zeros(
            minibatch_size, 3, 25, max_seqlength
        )  # 初始化一个全0的 长度为max_seqlength的矩阵
        lengths = torch.IntTensor(minibatch_size)
        targets = []
        input_size_orig = []
        phoneids = []
        dvsample = torch.zeros(
            minibatch_size, 3, batch[0][0].size(1), batch[0][0].size(2)
        )
        zssample = torch.zeros(
            minibatch_size, 3, batch[0][1].size(1), batch[0][1].size(2)
        )
        for x in range(minibatch_size):
            sample = batch[x]
            tensor = sample[2]
            input_size_ori = tensor.size(2)
            target = sample[3]
            inputs[x].narrow(2, 0, input_size_ori).copy_(tensor)  # 填充进全0矩阵
            targets.append(target)
            input_size_orig.append(input_size_ori / max_seqlength)
            dvsample[x] = batch[x][0]
            zssample[x] = batch[x][1]

        label = torch.tensor(targets, dtype=torch.long)
        return dvsample, zssample, inputs, label, np.array(input_size_orig)


class testpallmediaMyDataset(data.Dataset):
    def __init__(
        self, args, dvroot_dir, zsroot_dir, lroot_dir, islongimg, train, rou, padding=0
    ):
        super().__init__()
        self.size = {
            "media1": 234,
            "media2": 286,
            "media3": 44,
            "media4": 225,
            "media5": 186,
            "media6": 64,
            "media7": 80,
            "media8": 83,
            "media9": 152,
            "media10": 128,
            "media11": 135,
            "media12": 58,
            "media13": 75,
            "media14": 86,
        }

        w = int(720 * 1.2)
        h = int(576 * 1.2)
        self.dvroot_dir = dvroot_dir
        self.zsroot_dir = zsroot_dir
        self.lroot_dir = lroot_dir
        self.train = train
        self.islong = islongimg
        self.padding = padding
        self.round = rou
        setup_seed(3407)

        trainset = []
        for i in range(1, 15):
            trainsetm = os.listdir(f"{lroot_dir}/media{i}")
            for tm in trainsetm:
                trainset.append(["media" + str(i), tm])

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([224, 224]),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        if self.padding == 0:
            ltransform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize([50, 2000]),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            ltransform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(25),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        self.samples = trainset

        label = []
        for i in range(len(self.samples)):
            if self.samples[i][1][0] == "t":
                label.append(0)
            else:
                label.append(1)
        self.labels = label

        print(train)
        print(f"{len(self.samples)} samples")

        self.transform = transform
        self.ltransform = ltransform

        pos = np.sum(self.labels)
        neg = len(self.labels) - pos
        self.weights = torch.FloatTensor([1, neg / pos])

    def __len__(self):
        return len(self.samples)

    def getsamples(self):
        return self.samples

    def getlabel(self):
        return self.labels

    def __getitem__(self, index):
        dvarray = cv2.imread(
            f"{self.dvroot_dir}/{self.round}/{self.samples[index][0]}/{self.samples[index][1]}"
        )
        dvsample = self.transform(dvarray)
        zsarray = cv2.imread(
            f"{self.zsroot_dir}/{self.round}/{self.samples[index][0]}/{self.samples[index][1]}"
        )
        zssample = self.transform(zsarray)
        if self.islong == True:
            larray = cv2.imread(
                f"{self.lroot_dir}/{self.samples[index][0]}/{self.samples[index][1]}"
            )
            lsample = self.ltransform(larray)
        else:
            lsample = []
        # return sample,self.labels[index],self.size[self.samples[index][0]]
        # sample=self.images[index]
        label = self.labels[index]
        bz = self.size[self.samples[index][0]]

        return dvsample, zssample, lsample, label, bz, self.padding


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(3407)
