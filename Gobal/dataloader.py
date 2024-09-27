import os
import pandas as pd
import random

import numpy as np
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
    def __init__(self,args, root_dir,root_sample, roun, train, weights=None):
        super().__init__()
        self.size = {'media1': 234, 'media2': 286, 'media3': 44, 'media4': 225, 'media5': 186, 'media6': 64, 'media7': 80,
                'media8': 83, 'media9': 152, 'media10': 128, 'media11': 135, 'media12': 58, 'media13': 75,
                'media14': 86}

        w=int(720*1.2)
        h=int(576*1.2)
        self.round = roun
        self.root_dir = root_dir
        self.root_sample=root_sample
        self.train = train
        setup_seed(3407)


        fo = open(f'{self.root_sample}/participant14.txt', 'r')
        participants = []
        myFile = fo.read()
        myRecords = myFile.split('\n')
        for y in range(0, len(myRecords) - 1):
            participants.append(myRecords[y].split('\t'))  # 把表格数据存入

        if self.train=='test':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize([224, 224]),
                                            # transforms.CenterCrop(600),
                                            # transforms.Resize(100),
                                            # transforms.Grayscale(),
                                            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                            transforms.Normalize([0.5, 0.5,0.5], [0.5, 0.5,0.5])])
            # transforms.Normalize([0.485, ], [0.229, ])])
        else:

            transform = transforms.Compose([transforms.ToTensor(),
                                            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                            # transforms.RandomHorizontalFlip(p=0.5),
                                            # transforms.RandomVerticalFlip(p=0.5),
                                            transforms.Resize([224,224]),
                                            # transforms.CenterCrop(600),
                                            # transforms.Resize(100),
                                            # transforms.Grayscale(),
                                            # transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
                                            transforms.Normalize([0.5, 0.5,0.5], [0.5, 0.5,0.5])])
                                            # transforms.Normalize([0.485, ], [0.229, ])])

        trainset = []

        fo = open(f'{self.root_sample}/14/{self.round}/{train}sample.txt', 'r')
        myFile = fo.read()
        myRecords = myFile.split('\n')
        for y in range(0, len(myRecords) - 1):
            trainset.append(myRecords[y].split('\t'))  # 把表格数据存入

        self.samples = trainset

        label=[]
        for i in range(len(self.samples)):
            if self.samples[i][1][0]=='n':
                label.append(0)
            else:
                label.append(1)
        self.labels=label

        print(train)
        print(f'{len(self.samples)} samples')

        self.transform = transform

        pos = np.sum(self.labels)
        neg = len(self.labels) - pos
        self.weights = torch.FloatTensor([1, neg / pos])

        # seqlen=[]
        # videos=[]
        # images=[]
        # for index in range(len(self.samples)):
        #     array = cv2.imread(f'{self.root_dir}/{self.samples[index][0]}/{self.samples[index][1]}.jpg')
        #     array = self.transform(array)
        #     images.append(array)
        #     # print(index)
        #
        # self.images=images

    def __len__(self):
        return len(self.samples)

    def getsamples(self):
        return self.samples
    def getlabel(self):
        return self.labels

    def __getitem__(self, index):
        # files = os.listdir(f'{self.root_dir}/{self.samples[index][0]}/{self.samples[index][1]}')
        # files.sort(key=lambda x: int(x[5:-8]))
        # sample = torch.zeros(self.size['media2'], 1, 100, 100)
        # for i in range(len(files)):
        array = cv2.imread(f'{self.root_dir}/{self.round}/{self.samples[index][0]}/{self.samples[index][1]}.jpg')
        sample = self.transform(array)
        # return sample,self.labels[index],self.size[self.samples[index][0]]
        # sample=self.images[index]
        label=self.labels[index]

        return sample, label

class sapallmediaMyDataset(data.Dataset):
    def __init__(self,args, root_dir,rootsa,root_sample, roun, train, weights=None):
        super().__init__()
        self.size = {'media1': 234, 'media2': 286, 'media3': 44, 'media4': 225, 'media5': 186, 'media6': 64, 'media7': 80,
                'media8': 83, 'media9': 152, 'media10': 128, 'media11': 135, 'media12': 58, 'media13': 75,
                'media14': 86}

        w=int(720*1.2)
        h=int(576*1.2)
        self.round = roun
        self.root_dir = root_dir
        self.root_sample=root_sample
        self.rootsa=rootsa
        self.train = train
        setup_seed(3407)

        fo = open(f'{self.root_sample}/participant14.txt', 'r')
        participants = []
        myFile = fo.read()
        myRecords = myFile.split('\n')
        for y in range(0, len(myRecords) - 1):
            participants.append(myRecords[y].split('\t'))  # 把表格数据存入

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize([224,224]),
                                        transforms.Normalize([0.5, 0.5,0.5], [0.5, 0.5,0.5])])

        trainset = []

        fo = open(f'{self.root_sample}/14/{self.round}/{train}sample.txt', 'r')
        myFile = fo.read()
        myRecords = myFile.split('\n')
        for y in range(0, len(myRecords) - 1):
            trainset.append(myRecords[y].split('\t'))  # 把表格数据存入

        self.samples = trainset

        label=[]
        for i in range(len(self.samples)):
            if self.samples[i][1][0]=='n':
                label.append(0)
            else:
                label.append(1)
        self.labels=label

        print(train)
        print(f'{len(self.samples)} samples')

        self.transform = transform

        pos = np.sum(self.labels)
        neg = len(self.labels) - pos
        self.weights = torch.FloatTensor([1, neg / pos])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        saarry=cv2.imread(f'{self.rootsa}/{self.samples[index][0]}.jpg')
        array = cv2.imread(f'{self.root_dir}/{self.round}/{self.samples[index][0]}/{self.samples[index][1]}.jpg')
        sample = self.transform(array)
        sasample=self.transform(saarry)
        # return sample,self.labels[index],self.size[self.samples[index][0]]
        # sample=self.images[index]
        label=self.labels[index]

        return sample, label,sasample

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(3407)

# participants=['']
# for i in range(1,15):
#     if i !=3 and i!=6 and i!=12 and i!=14:
#         samples=os.listdir(f'../DataProcessing/gazepointimg/globalheatmap/media{i}')
#         for sample in samples:
#             bz=0
#             for j in range(len(participants)):
#                 if sample[:-5]==participants[j]:
#                     bz=1
#                     break
#             if bz==0:
#                 participants.append(sample[:-5])
# print(len(participants)-1)
# print(participants)
#
# ASD=[]
# TD=[]
# for i in range(1,len(participants)):
#     if participants[i][0]=='n':
#         TD.append(participants[i])
#     else:
#         ASD.append(participants[i])
#
# random.shuffle(ASD)
# random.shuffle(TD)
#
# # ASD14=ASD
# # ASD14.append('a010zxl')
#
# pASD=[ASD[:10],ASD[10:20],ASD[20:30],ASD[30:40],ASD[40:]]
# pASD14=[ASD[:10],ASD[10:20],ASD[20:30],ASD[30:40],ASD[40:]]
# pASD14[4].append('a010zxl')
# pTD=[TD[:7],TD[7:14],TD[14:21],TD[21:28],TD[28:]]
# print(pASD14)
# print(pTD)
#
# samples14 = []
# samples10=[]
# for i in range(1,15):
#    files=os.listdir(f'../DataProcessing/gazepointimg/globalheatmap/media{i}')
#    for file in files:
#        samples14.append(['media'+str(i),file[:-4]])
# for i in range(1,14):
#     if i!=3 and i!=6 and i!=12:
#        files=os.listdir(f'../DataProcessing/gazepointimg/globalheatmap/media{i}')
#        for file in files:
#            samples10.append(['media'+str(i),file[:-4]])
#
#
# for r in range(5):
#     train=[]
#     validation=[]
#     test=[]
#     if not os.path.exists(f'../sample/20220725/14/{r}'):
#         os.makedirs(f'../sample/20220725/14/{r}')
#     for k in range(len(samples14)):
#         for j in range(len(pASD14[r])//2):
#             if samples14[k][1][:-1]==pASD14[r][j]:
#                 validation.append(samples14[k])
#         for j in range(len(pASD14[r])//2,len(pASD14[r])):
#             if samples14[k][1][:-1] == pASD14[r][j]:
#                 test.append(samples14[k])
#         for j in range(5):
#             if j!=r:
#                 for m in range(len(pASD14[j])):
#                     if samples14[k][1][:-1] == pASD14[j][m]:
#                         train.append(samples14[k])
#         for j in range(len(pTD[r]) // 2):
#             if samples14[k][1][:-1] == pTD[r][j]:
#                 validation.append(samples14[k])
#         for j in range(len(pTD[r]) // 2, len(pTD[r])):
#             if samples14[k][1][:-1] == pTD[r][j]:
#                 test.append(samples14[k])
#         for j in range(5):
#             if j != r:
#                 for m in range(len(pTD[j])):
#                     if samples14[k][1][:-1] == pTD[j][m]:
#                         train.append(samples14[k])
#     # train = list(set([tuple(t) for t in train]))
#     # validation = list(set([tuple(t) for t in validation]))
#     # test = list(set([tuple(t) for t in test]))
#
#     print(len(train))
#     random.shuffle(train)
#     f = open(f'../sample/20220725/14/{r}/trainsample.txt','w')
#     for i in range(len(train)):
#         for j in range(len(train[i])):
#             f.write(train[i][j])
#             f.write('\t')
#         f.write('\n')
#     f.close()
#
#     f = open(f'../sample/20220725/14/{r}/validationsample.txt','w')
#     for i in range(len(validation)):
#         for j in range(len(validation[i])):
#             f.write(validation[i][j])
#             f.write('\t')
#         f.write('\n')
#     f.close()
#
#     f = open(f'../sample/20220725/14/{r}/testsample.txt','w')
#     for i in range(len(test)):
#         for j in range(len(test[i])):
#             f.write(test[i][j])
#             f.write('\t')
#         f.write('\n')
#     f.close()
#
# for r in range(5):
#     train=[]
#     validation=[]
#     test=[]
#     if not os.path.exists(f'../sample/20220725/10/{r}'):
#         os.makedirs(f'../sample/20220725/10/{r}')
#     for j in range(len(pASD[r])//2):
#         for k in range(len(samples10)):
#             if samples10[k][1][:-1]==pASD[r][j]:
#                 validation.append(samples10[k])
#     for j in range(len(pASD[r])//2,len(pASD[r])):
#         for k in range(len(samples10)):
#             if samples10[k][1][:-1] == pASD[r][j]:
#                 test.append(samples10[k])
#     for j in range(5):
#         if j!=r:
#             for k in range(len(pASD[j])):
#                 for m in range(len(samples10)):
#                     if samples10[m][1][:-1] == pASD[j][k]:
#                         train.append(samples10[k])
#     for j in range(len(pTD[r]) // 2):
#         for k in range(len(samples10)):
#             if samples10[k][1][:-1] == pTD[r][j]:
#                 validation.append(samples10[k])
#     for j in range(len(pTD[r]) // 2, len(pTD[r])):
#         for k in range(len(samples10)):
#             if samples10[k][1][:-1] == pTD[r][j]:
#                 test.append(samples10[k])
#     for j in range(5):
#         if j != r:
#             for k in range(len(pTD[j])):
#                 for m in range(len(samples10)):
#                     if samples10[m][1][:-1] == pTD[j][k]:
#                         train.append(samples10[m])
#     # train = list(set([tuple(t) for t in train]))
#     # validation = list(set([tuple(t) for t in validation]))
#     # test = list(set([tuple(t) for t in test]))
#
#     print(len(train))
#     random.shuffle(train)
#     f = open(f'../sample/20220725/10/{r}/trainsample.txt','w')
#     for i in range(len(train)):
#         for j in range(len(train[i])):
#             f.write(train[i][j])
#             f.write('\t')
#         f.write('\n')
#     f.close()
#
#     f = open(f'../sample/20220725/10/{r}/validationsample.txt','w')
#     for i in range(len(validation)):
#         for j in range(len(validation[i])):
#             f.write(validation[i][j])
#             f.write('\t')
#         f.write('\n')
#     f.close()
#
#     f = open(f'../sample/20220725/10/{r}/testsample.txt','w')
#     for i in range(len(test)):
#         for j in range(len(test[i])):
#             f.write(test[i][j])
#             f.write('\t')
#         f.write('\n')
#     f.close()
#
# p14=[]
# p10=[]
# for i in range(5):
#     p14i=pASD14[i]
#     p14i.extend(pTD[i])
#     p10i=pASD[i]
#     p10i.extend(pTD[i])
#     p14.append(p14i)
#     p10.append(p10i)
# # print(p14)
#
#
# if not os.path.exists('../sample/20220725'):
#     os.makedirs('../sample/20220725')
# f=open('../sample/20220725/participant14.txt','w')
# for i in range(len(p14)):
#     for j in range(len(p14[i])):
#         f.write(p14[i][j])
#         f.write('\t')
#     f.write('\n')
# f.close()
#
# f=open('../sample/20220725/participant10.txt','w')
# for i in range(len(p10)):
#     for j in range(len(p10[i])):
#         f.write(p10[i][j])
#         f.write('\t')
#     f.write('\n')
# f.close()


# participants=['']
# for i in range(1,15):
#     if i !=3 and i!=6 and i!=12 and i!=14:
#         samples=os.listdir(f'../DataProcessing/gazepointimg/globalheatmap/media{i}')
#         for sample in samples:
#             bz=0
#             for j in range(len(participants)):
#                 if sample[:-5]==participants[j]:
#                     bz=1
#                     break
#             if bz==0:
#                 participants.append(sample[:-5])
# print(len(participants)-1)
# print(participants)
#
# ASD=[]
# TD=[]
# for i in range(1,len(participants)):
#     if participants[i][0]=='n':
#         TD.append(participants[i])
#     else:
#         ASD.append(participants[i])
#
# random.shuffle(ASD)
# random.shuffle(TD)
# ASD.append('a010zxl')
# testASD=ASD[:8]
# testTD=TD[:5]
# p14=testASD+testTD
#
# if not os.path.exists('../sample/20220726/14'):
#     os.makedirs('../sample/20220726/14')
# f=open('../sample/20220726/participanttest.txt','w')
# for i in range(len(p14)):
#     f.write(p14[i])
#     f.write('\t')
# f.close()
#
# tvASD=ASD[8:]
# tvTD=TD[5:]
# tv=tvASD+tvTD
#
#
#
# psamples14 = []
# samples10=[]
# for i in range(1,15):
#    files=os.listdir(f'../DataProcessing/gazepointimg/globalheatmap/media{i}')
#    for j in range(len(p14)):
#        for file in files:
#            if p14[j]==file[:-5]:
#             psamples14.append(['media'+str(i),file[:-4]])
# f = open(f'../sample/20220726/14/testsample.txt','w')
# for i in range(len(psamples14)):
#     for j in range(len(psamples14[i])):
#         f.write(psamples14[i][j])
#         f.write('\t')
#     f.write('\n')
# f.close()
#
# tvsamplestd=[]
# tvsamplesasd=[]
# for i in range(1,15):
#     gdtd=[]
#     gdasd=[]
#     files=os.listdir(f'../DataProcessing/gazepointimg/globalheatmap/media{i}')
#     for file in files:
#         for j in range(len(tvASD)):
#             if file[:-5]==tvASD[j]:
#                 gdasd.append(['media'+str(i),file[:-4]])
#                 break
#         for j in range(len(tvTD)):
#             if file[:-5] == tvTD[j]:
#                 gdtd.append(['media' + str(i), file[:-4]])
#                 break
#     random.shuffle(gdtd)
#     random.shuffle(gdasd)
#     tvsamplesasd.append(gdasd)
#     tvsamplestd.append(gdtd)
#
# tvtd5=[]
# tvasd5=[]
# for i in range(14):
#     asd,td=[],[]
#     asdzs=len(tvsamplesasd[i])//5
#     tdzs=len(tvsamplestd[i])//5
#     asdys=len(tvsamplesasd[i])%5
#     tdys=len(tvsamplestd[i])%5
#     bz=0
#     for j in range(5-asdys):
#         asd.append(tvsamplesasd[i][int(asdzs*j):int(asdzs*(j+1))])
#         bz=int(asdzs*(i+1))
#     for j in range(5-asdys,5):
#         asd.append(tvsamplesasd[i][bz:int((asdzs+1) * (j + 1))])
#         bz=int((asdzs+1) * (j + 1))
#     tvasd5.append(asd)
#
#     bz = 0
#     for j in range(5 - tdys):
#         td.append(tvsamplestd[i][int(tdzs * j):int(tdzs * (j + 1))])
#         bz = int(tdzs * (i + 1))
#     for j in range(5 - tdys, 5):
#         td.append(tvsamplestd[i][bz:int((tdzs + 1) * (j + 1))])
#         bz = int((tdzs + 1) * (j + 1))
#     tvtd5.append(td)
#
# for i in range(5):
#     if not os.path.exists( '../sample/20220726/14/'+str(i)):
#         os.makedirs('../sample/20220726/14/'+str(i))
#     va=[]
#     tr=[]
#     for j in range(len(tvasd5)):
#         va.extend(tvasd5[j][i])
#     for j in range(len(tvtd5)):
#         va.extend(tvtd5[j][i])
#     f = open(f'../sample/20220726/14/{i}/validationsample.txt', 'w')
#     for j in range(len(va)):
#         for k in range(len(va[j])):
#             f.write(va[j][k])
#             f.write('\t')
#         f.write('\n')
#     f.close()
#     for j in range(len(tvasd5)):
#         for k in range(len(tvasd5[j])):
#             if k!=i:
#                 tr.extend(tvasd5[j][k])
#     for j in range(len(tvtd5)):
#         for k in range(len(tvtd5[j])):
#             if k!=i:
#                 tr.extend(tvtd5[j][k])
#     random.shuffle(tr)
#     f = open(f'../sample/20220726/14/{i}/trainsample.txt', 'w')
#     for j in range(len(tr)):
#         for k in range(len(tr[j])):
#             f.write(tr[j][k])
#             f.write('\t')
#         f.write('\n')
#     f.close()



# for i in range(5):
#     fo = open(f'../sample/20220725/14/{i}/trainsample.txt', 'r')
#     trainsample = []
#     myFile = fo.read()
#     myRecords = myFile.split('\n')
#     for y in range(0, len(myRecords) - 1):
#         trainsample.append(myRecords[y].split('\t'))  # 把表格数据存入
#
#     fo = open(f'../sample/20220725/14/{i}/validationsample.txt', 'r')
#     validationsample = []
#     myFile = fo.read()
#     myRecords = myFile.split('\n')
#     for y in range(0, len(myRecords) - 1):
#         validationsample.append(myRecords[y].split('\t'))  # 把表格数据存入
#
#     random.shuffle(validationsample)
#
#     trainsample.extend(validationsample)
#     f = open(f'../sample/20220725/14/{i}/trainvalidationsample.txt', 'w')
#     for j in range(len(trainsample)):
#         for k in range(len(trainsample[j])):
#             f.write(trainsample[j][k])
#             f.write('\t')
#         f.write('\n')
#     f.close()


###由participant生成sample
# fo = open('../sample/20220814/participant14.txt', 'r')
# participantsjc = []
# myFile = fo.read()
# myRecords = myFile.split('\n')
# for y in range(0, len(myRecords) - 1):
#     participantsjc.append(myRecords[y].split('\t'))  # 把表格数据存入
#
# fo = open('../sample/20220817/participant14.txt', 'r')
# participants = []
# myFile = fo.read()
# myRecords = myFile.split('\n')
# for y in range(0, len(myRecords) - 1):
#     participants.append(myRecords[y].split('\t'))  # 把表格数据存入
#
# for i in range(len(participantsjc)):
#     for j in range(len(participantsjc[i])-1):
#         e=0
#         for m in range(len(participants)):
#             for n in range(len(participants[m])-1):
#                 if participants[m][n]==participantsjc[i][j]:
#                     e=1
#         if e==0:
#             print(participantsjc[i][j])
#
# for i in range(5):
#     if not os.path.exists( '../sample/20220817/14/'+str(i)):
#         os.makedirs('../sample/20220817/14/'+str(i))
#     tv = []
#     test = []
#     for j in range(1,15):
#         files=os.listdir(f'../DataProcessing/gazepointimg/globalheatmap/media{j}')
#         for file in files:
#             for k in range(len(participants[i])):
#                 if file[:-5]==participants[i][k]:
#                     test.append(['media'+str(j),file[:-4]])
#                     break
#             for k in range(len(participants)):
#                 if k!=i:
#                     for m in range(len(participants[k])):
#                         if file[:-5] == participants[k][m]:
#                             tv.append(['media' + str(j), file[:-4]])
#                             break
#         random.shuffle(tv)
#
#     f = open(f'../sample/20220817/14/{i}/trainvalidationsample.txt', 'w')
#     for j in range(len(tv)):
#         for k in range(len(tv[j])):
#             f.write(tv[j][k])
#             f.write('\t')
#         f.write('\n')
#     f.close()
#
#     f = open(f'../sample/20220817/14/{i}/testsample.txt', 'w')
#     for j in range(len(test)):
#         for k in range(len(test[j])):
#             f.write(test[j][k])
#             f.write('\t')
#         f.write('\n')
#     f.close()
###

# fo = open('../sample/20220814/participant14.txt', 'r')
# ASD,TD=[],[]
# participants = []
# myFile = fo.read()
# myRecords = myFile.split('\n')
# for y in range(0, len(myRecords) - 1):
#     participants.append(myRecords[y].split('\t'))  # 把表格数据存入
# for i in range(len(participants)):
#     for j in range(len(participants[i])-1):
#         if participants[i][j][0]=='n':
#             TD.append(participants[i][j])
#         else:
#             ASD.append(participants[i][j])
#
# all=ASD+TD
# for i in range(len(all)):
#     record=0
#     for j in range(1,15):
#         files=os.listdir(f'../DataProcessing/gazepointimg/gazepointmap15/media{j}')
#         for file in files:
#             if file[:-5]==all[i]:
#                 record+=1
#     print(all[i],record)
# print(len(all))

# sequence=pd.read_excel('../../数据20210623/视频播放顺序.xlsx')
# sequence=sequence.values.tolist()
# sequence=np.delete(sequence,[37,38],0)
# s4=[]
# for i in range(len(sequence)):
#     for j in range(len(sequence[i])):
#         if sequence[i][j]==str(4.0):
#             s4.append(sequence[i][0]+str(j))
# for i in range(5):
#     if not os.path.exists(f'../sample/20220817wos4/14/{i}'):
#         os.makedirs(f'../sample/20220817wos4/14/{i}')
#     newtrain,newtest=[],[]
#     fo = open(f'../sample/20220817/14/{i}/trainvalidationsample.txt', 'r')
#     train,test=[],[]
#     myFile = fo.read()
#     myRecords = myFile.split('\n')
#     for y in range(0, len(myRecords) - 1):
#         train.append(myRecords[y].split('\t'))  # 把表格数据存入
#
#     fo = open(f'../sample/20220817/14/{i}/testsample.txt', 'r')
#     myFile = fo.read()
#     myRecords = myFile.split('\n')
#     for y in range(0, len(myRecords) - 1):
#         test.append(myRecords[y].split('\t'))  # 把表格数据存入
#
#     for j in range(len(train)):
#         e=0
#         for k in range(len(s4)):
#             if train[j][1]==s4[k]:
#                 e=1
#         if e==0:
#             newtrain.append(train[j])
#
#     for j in range(len(test)):
#         e=0
#         for k in range(len(s4)):
#             if test[j][1]==s4[k]:
#                 e=1
#         if e==0:
#             newtest.append(test[j])
#
#     f=open(f'../sample/20220817wos4/14/{i}/trainvalidationsample.txt','w')
#     for j in range(len(newtrain)):
#         for k in range(len(newtrain[j])):
#             f.write(newtrain[j][k])
#             f.write('\t')
#         f.write('\n')
#     f.close()
#
#     f = open(f'../sample/20220817wos4/14/{i}/testsample.txt', 'w')
#     for j in range(len(newtest)):
#         for k in range(len(newtest[j])):
#             f.write(newtest[j][k])
#             f.write('\t')
#         f.write('\n')
#     f.close()

# sample=[]
# fo = open(f'../sample/20220817wos4/14/0/testsample.txt', 'r')
# myFile = fo.read()
# myRecords = myFile.split('\n')
# for y in range(0, len(myRecords) - 1):
#     sample.append(myRecords[y].split('\t'))  # 把表格数据存入
# fo = open(f'../sample/20220817wos4/14/0/trainvalidationsample.txt', 'r')
# myFile = fo.read()
# myRecords = myFile.split('\n')
# for y in range(0, len(myRecords) - 1):
#     sample.append(myRecords[y].split('\t'))  # 把表格数据存入
# asd,td=0,0
# for i in range(len(sample)):
#     if sample[i][1][0]=='n':
#         td+=1
#     else:
#         asd+=1
# print(asd,td)