import os
import pandas as pd
import random

import numpy as np
import numba
from numba import jit
import gc
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data
from torchvision import transforms

size = {
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


class pallmediaMyDataset(data.Dataset):
    '''
    
    '''
    def __init__(self, root_dir, root_sample, roun, train):
            """
            初始化数据加载器类。

            参数：
            - root_dir：数据根目录
            - root_sample：样本根目录
            - roun：轮次
            - train：训练类型

            属性：
            - size：媒体大小字典
            - round：轮次
            - root_dir：数据根目录
            - root_sample：样本根目录
            - train：训练类型
            - samples：样本列表
            - labels：标签列表
            - transform：数据转换器
            - weights：样本权重
            - maxlength：最大长度
            """
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
            self.root_dir = root_dir
            self.root_sample = root_sample
            self.train = train

            fo = open(f"{self.root_sample}/participant14.txt", "r")
            participants = []
            myFile = fo.read()
            myRecords = myFile.split("\n")
            for y in range(0, len(myRecords) - 1):
                participants.append(myRecords[y].split("\t"))  # 把表格数据存入

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(25),
                    # transforms.Grayscale(),
                    # transforms.Normalize([0.485, ], [0.229, ])])
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            # transforms.Normalize([0.5, ], [0.5, ])])

            trainset = []

            fo = open(f"{self.root_sample}/14/{self.round}/{train}sample.txt", "r")
            myFile = fo.read()
            myRecords = myFile.split("\n")
            for y in range(0, len(myRecords) - 1):
                trainset.append(myRecords[y].split("\t"))  # 把表格数据存入

            self.samples = trainset

            label = []
            for i in range(len(self.samples)):
                if self.samples[i][1][0] == "n":
                    label.append(0)
                else:
                    label.append(1)
            self.labels = label

            print(train)
            print(f"{len(self.samples)} samples")

            self.transform = transform

            pos = np.sum(self.labels)
            neg = len(self.labels) - pos
            self.weights = torch.FloatTensor([1, neg / pos])
            self.maxlength = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        array = Image.open(
            f"{self.root_dir}/{self.samples[index][0]}/{self.samples[index][1]}.jpg"
        )
        label = self.labels[index]
        sample = self.samples[index]

        array = self.transform(array)
        # b0=torch.zeros((3,50,int((self.size['media2']-self.size[sample[0]])*50)))
        # array=torch.cat([array,b0],dim=2)
        bz = self.size[sample[0]]

        return array, label, bz


class cutpallmediaMyDataset(data.Dataset):
    def __init__(self, root_dir, root_sample, roun, train):
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
        self.root_dir = root_dir
        self.root_sample = root_sample
        self.train = train

        fo = open(f"{self.root_sample}/participant14.txt", "r")
        participants = []
        myFile = fo.read()
        myRecords = myFile.split("\n")
        for y in range(0, len(myRecords) - 1):
            participants.append(myRecords[y].split("\t"))  # 把表格数据存入

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((50, 2000)),
                # transforms.Grayscale(),
                # transforms.Normalize([0.485, ], [0.229, ])])
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        # transforms.Normalize([0.5, ], [0.5, ])])

        trainset = []

        fo = open(f"{self.root_sample}/14/{self.round}/{train}sample.txt", "r")
        myFile = fo.read()
        myRecords = myFile.split("\n")
        for y in range(0, len(myRecords) - 1):
            trainset.append(myRecords[y].split("\t"))  # 把表格数据存入

        cuttrainset = []
        for y in range(len(trainset)):
            names = os.listdir(f"{self.root_dir}/{trainset[y][0]}")
            for name in names:
                if name[: len(trainset[y][1])] == trainset[y][1]:
                    cuttrainset.append([trainset[y][0], name])

        self.samples = cuttrainset

        label = []
        for i in range(len(self.samples)):
            if self.samples[i][1][0] == "n":
                label.append(0)
            else:
                label.append(1)
        self.labels = label

        print(train)
        print(f"{len(self.samples)} samples")

        self.transform = transform

        pos = np.sum(self.labels)
        neg = len(self.labels) - pos
        self.weights = torch.FloatTensor([1, neg / pos])
        self.maxlength = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        array = Image.open(
            f"{self.root_dir}/{self.samples[index][0]}/{self.samples[index][1]}"
        )
        label = self.labels[index]
        sample = self.samples[index]

        array = self.transform(array)
        # b0=torch.zeros((3,50,int((self.size['media2']-self.size[sample[0]])*50)))
        # array=torch.cat([array,b0],dim=2)
        bz = self.size[sample[0]]

        return array, label, bz


def _collate_fn(batch):
    def func(p):
        return p[0].size(2)

    # batch = sorted(batch, key=lambda sample: sample[0].size(2), reverse=True)

    longest_sample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(2)
    inputs = torch.zeros(
        minibatch_size, 3, 25, max_seqlength
    )  # 初始化一个全0的 长度为max_seqlength的矩阵
    lengths = torch.IntTensor(minibatch_size)
    targets = []
    input_size_orig = []
    phoneids = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        input_size_ori = tensor.size(2)
        target = sample[1]
        inputs[x].narrow(2, 0, input_size_ori).copy_(tensor)  # 填充进全0矩阵
        targets.append(target)
        input_size_orig.append(input_size_ori / max_seqlength)

    label = torch.tensor(targets, dtype=torch.long)
    return inputs, label, np.array(input_size_orig)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(3407)

# transform = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Grayscale(),
#                                         transforms.Normalize([0.485, ], [0.229, ])])
# img=cv2.imread('035hyc3.jpg')
# # img1=t(img)
# # print(img1)
# img=transform(img)
# print(img)
# b0=torch.zeros((1,50,5))
# im=torch.cat([img,b0],dim=2)
# print(im)
# print(img.shape)
# print(im.shape)
