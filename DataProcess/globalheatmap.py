import random

import numpy as np
import pandas as pd
import os
import xlwt
from multiprocessing import Process
import openpyxl
import cv2
import matplotlib.pyplot as plt
import numpy.matlib as mb
import scipy

"""
生成注意力图像
gazepointmapmm(paramas,sca,kernalsize,media)

"""


class paramas:
    def __init__(self):
        self.datafile = "20210623"
        self.savefile = "1datadecolumn"

        self.savesaccadecoor = "2saccadecoor"
        self.savegazepointcoor = "2gazepointcoor"

        self.savesaccadefixation = "3saccadefixation"
        self.savefixation = "3fixation"
        self.savegazepoint = "3gazepoint"
        self.videostart = "videostart"
        self.crudegazepoint = "3crudegazepoint"

        self.oksaccadefixation = "5oksaccadefixation"
        self.okfixation = "5okfixation"
        self.okgazepoint = "5okgazepoint"
        self.okcrudegazepoint = "5okcrudegazepoint"
        self.videostartsplit = "videostartsplit"

        # 处理完的一些样本是不能用的，能用的样本在这里
        self.fixationsample = "6fixationsample"
        self.saccadefixationsample = "6saccadefixationsample"
        self.gazepointsample = "6gazepointsample"
        self.crudegazepointsample = "6crudegazepointsample"

        self.fixationimgcoor = "fixationimg/fixationcoor"
        self.saccadefixationimgcoor = "saccadefixationimg/fixationcoor"
        self.gazepointimgcoor = "gazepointimg/fixationcoor"
        self.crudegazepointimgcoor = "gazepointimg/crudegazepoint"
        self.delecrudegazepointimgcoor = "gazepointimg/delecrudegazepoint"  #"gazepointimg_JY/delecrudegazepoint_JY"
        self.gazepointfps = "gazepointfps"

        self.globalheatmap = "gazepointimg/globalheatmap"
        self.allheatmap = "gazepointimg/allheatmap"
        self.gazepointmap = "gazepointimg/gazepointmap"
        self.gazepointmapmm = "gazepointimg/gazepointimgmmwos4"
        self.addgauss = "gazepointimg/addgauss"
        self.root_sample = "../sample/20220817wos4/14"


zs_normalize = {
    0: {
        1: 9.0,
        2: 12.0,
        3: 8.0,
        4: 14.0,
        5: 13.0,
        6: 10.0,
        7: 10.0,
        8: 6.0,
        9: 11.0,
        10: 9.0,
        11: 10.0,
        12: 10.0,
        13: 7.0,
        14: 10.0,
    },
    1: {
        1: 9.0,
        2: 11.0,
        3: 8.0,
        4: 14.0,
        5: 11.0,
        6: 8.0,
        7: 10.0,
        8: 9.0,
        9: 12.0,
        10: 9.0,
        11: 10.0,
        12: 10.0,
        13: 7.0,
        14: 10.0,
    },
    2: {
        1: 9.0,
        2: 12.0,
        3: 8.0,
        4: 14.0,
        5: 13.0,
        6: 10.0,
        7: 10.0,
        8: 9.0,
        9: 12.0,
        10: 9.0,
        11: 8.0,
        12: 9.0,
        13: 7.0,
        14: 9.0,
    },
    3: {
        1: 9.0,
        2: 12.0,
        3: 7.0,
        4: 10.0,
        5: 13.0,
        6: 10.0,
        7: 10.0,
        8: 9.0,
        9: 12.0,
        10: 9.0,
        11: 10.0,
        12: 10.0,
        13: 7.0,
        14: 10.0,
    },
    4: {
        1: 9.0,
        2: 12.0,
        3: 8.0,
        4: 14.0,
        5: 13.0,
        6: 10.0,
        7: 10.0,
        8: 9.0,
        9: 12.0,
        10: 8.0,
        11: 10.0,
        12: 10.0,
        13: 7.0,
        14: 10.0,
    },
}


def maxminnormalize(paramas, sca, rou, media):
    fo = open(f"{paramas.root_sample}/{rou}/trainvalidationsample.txt", "r")
    participants = []
    myFile = fo.read()
    myRecords = myFile.split("\n")
    max = 0
    for y in range(0, len(myRecords) - 1):
        participants.append(myRecords[y].split("\t"))  # 把表格数据存入
    for i in range(len(participants)):
        if participants[i][0] == "media" + str(media):
            sheet = pd.read_excel(
                f"{paramas.delecrudegazepointimgcoor}{sca}/media{media}/{participants[i][1]}.xlsx"
            )
            sheet = sheet.values.tolist()
            jz = np.zeros((576, 720))
            for i in range(len(sheet)):
                if str(sheet[i][2]) != "nan":
                    x = round(float(sheet[i][2]) * 720)
                    y = round(float(sheet[i][3]) * 576)
                    jz[y - 1][x - 1] += 1
            if np.max(jz) > max:
                max = np.max(jz)
    print(f"{media}:{max},")
    return max


# 最大值滤波
def max_box(image, kernalsize):
    mean_image = np.zeros(shape=image.shape, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            mean_image[i][j] = np.max(image[i : i + kernalsize, j : j + kernalsize])
    return mean_image


def gazepointmapmm(paramas, sca, kernalsize, media):
    for r in range(5):
        BL = zs_normalize[r][media]
        if not os.path.exists(f"{paramas.gazepointmapmm}{kernalsize}/{r}/media{media}"):
            os.makedirs(f"{paramas.gazepointmapmm}{kernalsize}/{r}/media{media}")

        samples = os.listdir(f"{paramas.delecrudegazepointimgcoor}{sca}/media{media}")
        for sample in samples:
            sheet = pd.read_excel(
                f"{paramas.delecrudegazepointimgcoor}{sca}/media{media}/{sample}"
            )
            sheet = sheet.values.tolist()
            jz = np.zeros((576, 720))
            for i in range(len(sheet)):
                if str(sheet[i][2]) != "nan":
                    x = round(float(sheet[i][2]) * 720)
                    y = round(float(sheet[i][3]) * 576)
                    jz[y - 1][x - 1] += 1
            # BL=np.max(jz)
            jz = jz / BL * 255
            jz[jz > 255] = 255
            heat_img = max_box(jz, kernalsize)
            cv2.imwrite(
                f"{paramas.gazepointmapmm}{kernalsize}/{r}/media{media}/{sample[:-5]}.jpg",
                heat_img,
            )


pa = paramas()
process_list = []

for i in range(1, 15):  # 开启5个子进程执行fun1函数
    # if i!=14:
    p = Process(target=gazepointmapmm, args=(pa, 0.5, 15, i))  # 实例化进程对象
    p.start()
    process_list.append(p)

for i in process_list:
    i.join()


# np.set_printoptions(threshold=np.inf)
# globalheatmap(pa,0.5,1)

# allheatmap(pa,0.5)

# gazepointmap(pa,0.5,10,1)

# addguass(pa,0.5,1)


"""
def globalheatmap(paramas,sca,media):
    if not os.path.exists(f'{paramas.globalheatmap}/media{media}'):
        os.makedirs(f'{paramas.globalheatmap}/media{media}')

    samples=os.listdir(f'{paramas.delecrudegazepointimgcoor}{sca}/media{media}')
    for sample in samples:
        sheet=pd.read_excel(f'{paramas.delecrudegazepointimgcoor}{sca}/media{media}/{sample}')
        sheet=sheet.values.tolist()
        jz=np.zeros((576,720))
        for i in range(len(sheet)):
            if str(sheet[i][2])!='nan':
                x=round(float(sheet[i][2])*720)
                y=round(float(sheet[i][3])*576)
                jz[y-1][x-1]+=1
        # print(jz)
        BL=np.max(jz)
        # gray=np.ones((576,720))
        # for i in range(len(jz)):
        #     for j in range(len(jz[i])):
        #         if jz[i][j]!=0 and jz[i][j]!=BL:
        #             Tij=jz[i][j]/BL*gauss(576,720,j,i)
        #             gray=gray+Tij
        # print(gray)
        # gray[gray>1]=1
        # print(gray)
        jz=jz/BL
        gray = cv2.GaussianBlur(jz, (75, 75), 37.5)
        norm_img = np.zeros(gray.shape)
        cv2.normalize(gray, norm_img, 0, 255, cv2.NORM_MINMAX)
        norm_img = np.asarray(norm_img, dtype=np.uint8)
        # print(norm_img)
        heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_RAINBOW)  # 注意此处的三通道热力图是cv2专有的GBR排列
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
        cv2.imwrite(f'{paramas.globalheatmap}/media{media}/{sample[:-5]}.jpg',heat_img)

def allheatmap(paramas,sca):
    if not os.path.exists(f'{paramas.allheatmap}'):
        os.makedirs(f'{paramas.allheatmap}')

    asdjz = np.zeros((576, 720))
    tdjz = np.zeros((576, 720))
    for media in range(1,15):
        samples=os.listdir(f'{paramas.delecrudegazepointimgcoor}{sca}/media{media}')
        for sample in samples:
            sheet=pd.read_excel(f'{paramas.delecrudegazepointimgcoor}{sca}/media{media}/{sample}')
            sheet=sheet.values.tolist()
            if sample[0]=='n':
                for i in range(len(sheet)):
                    if str(sheet[i][2])!='nan':
                        x=round(float(sheet[i][2])*720)
                        y=round(float(sheet[i][3])*576)
                        tdjz[y - 1][x - 1]+=1
            else:
                for i in range(len(sheet)):
                    if str(sheet[i][2])!='nan':
                        x=round(float(sheet[i][2])*720)
                        y=round(float(sheet[i][3])*576)
                        asdjz[y - 1][x - 1]+=1

    asdBL=np.max(asdjz)
    asdjz= asdjz / asdBL
    asdgray = cv2.GaussianBlur(asdjz, (75, 75), 37.5)
    asdnorm_img = np.zeros(asdgray.shape)
    cv2.normalize(asdgray, asdnorm_img, 0, 255, cv2.NORM_MINMAX)
    asdnorm_img = np.asarray(asdnorm_img, dtype=np.uint8)
    # print(norm_img)
    asdheat_img = cv2.applyColorMap(asdnorm_img, cv2.COLORMAP_RAINBOW)  # 注意此处的三通道热力图是cv2专有的GBR排列
    asdheat_img = cv2.cvtColor(asdheat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
    cv2.imwrite(f'{paramas.allheatmap}/ASD.jpg', asdheat_img)

    tdBL = np.max(tdjz)
    tdjz = tdjz / tdBL
    tdgray = cv2.GaussianBlur(tdjz, (75, 75), 37.5)
    tdnorm_img = np.zeros(tdgray.shape)
    cv2.normalize(tdgray, tdnorm_img, 0, 255, cv2.NORM_MINMAX)
    tdnorm_img = np.asarray(tdnorm_img, dtype=np.uint8)
    # print(norm_img)
    tdheat_img = cv2.applyColorMap(tdnorm_img, cv2.COLORMAP_RAINBOW)  # 注意此处的三通道热力图是cv2专有的GBR排列
    tdheat_img = cv2.cvtColor(tdheat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
    cv2.imwrite(f'{paramas.allheatmap}/TD.jpg', tdheat_img)


def gauss(IMAGE_HEIGHT, IMAGE_WIDTH, center_x, center_y):  # provide gauss RGB map

    #R = np.sqrt(center_x ** 2 + center_y ** 2)
    R= 25
    # R=15

    Gauss_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

    # 直接利用矩阵运算实现

    mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)

    mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)

    x1 = np.arange(IMAGE_WIDTH)

    x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)

    y1 = np.arange(IMAGE_HEIGHT)

    y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)

    y_map = np.transpose(y_map)

    Gauss_map = np.sqrt((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2)

    Gauss_map = np.exp(-0.5 * Gauss_map / R)  #
    return Gauss_map

def gazepointmap(paramas,sca,kernalsize,media):
    if not os.path.exists(f'{paramas.gazepointmap}{kernalsize}/media{media}'):
        os.makedirs(f'{paramas.gazepointmap}{kernalsize}/media{media}')

    samples=os.listdir(f'{paramas.delecrudegazepointimgcoor}{sca}/media{media}')
    for sample in samples:
        sheet=pd.read_excel(f'{paramas.delecrudegazepointimgcoor}{sca}/media{media}/{sample}')
        sheet=sheet.values.tolist()
        jz=np.zeros((576,720))
        for i in range(len(sheet)):
            if str(sheet[i][2])!='nan':
                x=round(float(sheet[i][2])*720)
                y=round(float(sheet[i][3])*576)
                jz[y-1][x-1]+=1
        BL=np.max(jz)
        jz=jz/BL*255
        heat_img=max_box(jz,kernalsize)
        cv2.imwrite(f'{paramas.gazepointmap}{kernalsize}/media{media}/{sample[:-5]}.jpg',heat_img)


def addguass(paramas,sca,media):
    if not os.path.exists(f'{paramas.addgauss}/media{media}'):
        os.makedirs(f'{paramas.addgauss}/media{media}')

    samples=os.listdir(f'{paramas.delecrudegazepointimgcoor}{sca}/media{media}')
    for sample in samples:
        sheet=pd.read_excel(f'{paramas.delecrudegazepointimgcoor}{sca}/media{media}/{sample}')
        sheet=sheet.values.tolist()
        jz=np.zeros((576,720))
        gaussmap=np.zeros((576,720))

        for i in range(len(sheet)):
            e=0
            if str(sheet[i][2])!='nan':
                e+=1
                x=round(float(sheet[i][2])*720)
                y=round(float(sheet[i][3])*576)
                jz[y-1][x-1]+=1
                gaussmap=gaussmap+np.sqrt(gauss(576,720,x,y))
                # if e>2:
                #     break
        BL=np.max(jz)
        jz=jz/BL
        # gaussmap=gaussmap/e
        # gray = cv2.GaussianBlur(jz, (75, 75), 37.5)
        norm_img = np.zeros(gaussmap.shape)
        cv2.normalize(gaussmap, norm_img, 0, 255, cv2.NORM_MINMAX)
        norm_img = np.asarray(norm_img, dtype=np.uint8)
        # print(norm_img)
        heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_RAINBOW)  # 注意此处的三通道热力图是cv2专有的GBR排列
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
        cv2.imwrite(f'{paramas.addgauss}/media{media}/{sample[:-5]}.jpg',heat_img)    
"""
