import numpy as np
import os
import pandas as pd
import cv2
import openpyxl
from multiprocessing import Process
import xlwt
from numba import jit
from numba.typed import List
from PIL import Image

'''
生成视频的整体显著性图像并将显著性图像和注意力图像预设值一定的权重进行叠加

按顺序执行
1.生成视频的整体显著性图像   addsaliency(paramas,视频编号)
处理后的文件保存在paramas.saliencyadd='saliencyadd'
2.将显著性图像和注意力图像预设值一定的权重进行叠加   gazeaddsaliencymm(paramas,0.5,视频编号)
处理后的文件保存在paramas.saliencyaddgazemm = 'saliencyaddgazemmwos4'
'''

size = {'media1': 234, 'media2': 286, 'media3':44,'media4':225, 'media5': 186, 'media6':64,'media7': 80,
            'media8': 83, 'media9': 152,'media10': 128, 'media11': 135,'media12':58, 'media13':75,'media14':86}

class paramas():
    def __init__(self):
        self.datafile = '20210623'
        self.savefile = '1datadecolumn'

        self.savesaccadecoor = '2saccadecoor'
        self.savegazepointcoor = '2gazepointcoor'

        self.savesaccadefixation = '3saccadefixation'
        self.savefixation = '3fixation'
        self.savegazepoint = '3gazepoint'
        self.videostart = 'videostart'

        self.oksaccadefixation = '5oksaccadefixation'
        self.okfixation = '5okfixation'
        self.okgazepoint = '5okgazepoint'
        self.videostartsplit = 'videostartsplit'

        self.fixationsample = '6fixationsample'
        self.saccadefixationsample = '6saccadefixationsample'
        self.gazepointsample = '6gazepointsample'

        self.fixationimgcoor = 'fixationimg/fixationcoor'
        self.saccadefixationimgcoor = 'saccadefixationimg/fixationcoor'
        self.gazepointimgcoor = 'gazepointimg/fixationcoor'
        self.gazepointfps = 'gazepointfps'

        self.fixationfsjoint = 'fixationimg/fsjoint14ss'
        self.saccadefixationfsjoint = 'saccadefixationimg/fsjoint10ss'
        self.gazepointfsjoint = 'gazepointimg/fsjoint14ss'
        self.newfixationfsjoint = 'newfixationimg/fsjoint10ss'
        self.newsaccadefixationfsjoint = 'newsaccadefixationimg/fsjoint10ss'
        self.newgazepointfsjoint = 'newgazepointimg/fsjoint10ss'
        self.saliency='../Saliency'
        self.newsaliency='../TASED-Net-master/output'
        self.frame='../img/frame'
        self.gazepointfsjoint1d='gazepointimg/fsjoint1dcrude'
        self.newgazepointfsjoint1d = 'newgazepointimg/fsjoint1d'
        self.fixationfsjoint1d='fixationimg/fsjoint1d'
        self.newfixationfsjoint1d='newfixationimg/fsjoint1d'
        self.saccadefixationfsjoint1d='saccadefixationimg/fsjoint1d'
        self.newsaccadefixationfsjoint1d='newsaccadefixationimg/fsjoint1d'
        self.videosaliencyhighlight='gazepointimg/9videosaliencyhighlight'
        self.sequence='gazepointimg/sequence'
        self.processvideosaliencyhighlight='gazepointimg/p9videosaliencyhighlight80'
        self.saliencyadd='saliencyadd'
        self.saliencyaddgaze='saliencyaddgaze'
        self.saliencyaddgazergb = 'saliencyaddgazergb'
        self.gazepointmap = 'gazepointimg/gazepointmap'
        self.gazepointmapmm = 'gazepointimg/gazepointimgmmwos4'
        self.saliencyaddgazemm = 'saliencyaddgazemmwos4'
        self.saliencyaddgazergbmm = 'saliencyaddgazemmwos4rgb'

def addsaliency(paramas,media):
    frames=os.listdir(f'{paramas.saliency}/media{media}')
    img=np.zeros((576,720,3))
    length=size['media'+str(media)]
    for frame in frames:
        jz=cv2.imread(f'{paramas.saliency}/media{media}/{frame}')
        img=img+jz/length
    cv2.imwrite(f'{paramas.saliencyadd}/media{media}.jpg',img)

def gazeaddsaliencymm(paramas,sca,media):
    for r in range(5):
        if not os.path.exists(f'{paramas.saliencyaddgazemm}/{r}/media{media}'):
            os.makedirs(f'{paramas.saliencyaddgazemm}/{r}/media{media}')
        samples=os.listdir(f'{paramas.gazepointmapmm}{sca}/{r}/media{media}')
        saliencyadd=cv2.imread(f'{paramas.saliencyadd}/media{media}.jpg')
        for sample in samples:
            img=cv2.imread(f'{paramas.gazepointmapmm}{sca}/{r}/media{media}/{sample}')
            img1=cv2.addWeighted(saliencyadd,0.4,img,0.6,0)
            cv2.imwrite(f'{paramas.saliencyaddgazemm}/{r}/media{media}/{sample}',img1)

def dveaddsaliencymm(paramas,media):
    for r in range(5):
        if not os.path.exists(f'{paramas.saliencyaddgazemm}/{r}/media{media}'):
            os.makedirs(f'{paramas.saliencyaddgazemm}/{r}/media{media}')
        samples=os.listdir(f'{paramas.gazepointmapmm}/{r}/media{media}')
        saliencyadd=cv2.imread(f'{paramas.saliencyadd}/media{media}.jpg')
        for sample in samples:
            img=cv2.imread(f'{paramas.gazepointmapmm}/{r}/media{media}/{sample}')
            img1=cv2.addWeighted(saliencyadd,0,img,1,0)
            cv2.imwrite(f'{paramas.saliencyaddgazemm}/{r}/media{media}/{sample}',img1)

def gazeaddsaliencymmrgb(paramas,sca,ro,media):
    for r in range(5):
        if not os.path.exists(f'{paramas.saliencyaddgazergbmm}{ro}/{r}/media{media}'):
            os.makedirs(f'{paramas.saliencyaddgazergbmm}{ro}/{r}/media{media}')
        samples=os.listdir(f'{paramas.gazepointmapmm}{sca}/{r}/media{media}')
        saliencyadd=cv2.imread(f'{paramas.saliencyadd}/media{media}.jpg')
        G=saliencyadd[:,:,0]*0.5
        G=G.astype(int)
        # print(G.shape)
        G=G.astype(np.uint8)
        # print(G)
        # print(G.dtype)
        for sample in samples:
            R=np.full((576,720), ro, dtype=np.uint8)
            # print(R.dtype)
            img=cv2.imread(f'{paramas.gazepointmapmm}{sca}/{r}/media{media}/{sample}')
            # print(img)
            B=img[:,:,0]
            # B.astype(np.float64)
            # print(B.dtype)
            merged = cv2.merge([B,G,R])
            cv2.imwrite(f'{paramas.saliencyaddgazergbmm}{ro}/{r}/media{media}/{sample}', merged)

def gazeaddsaliencyrgb(paramas,sca,r,media):
    if not os.path.exists(f'{paramas.saliencyaddgazergb}{r}/media{media}'):
        os.makedirs(f'{paramas.saliencyaddgazergb}{r}/media{media}')
    samples=os.listdir(f'{paramas.gazepointmap}{sca}/media{media}')
    saliencyadd=cv2.imread(f'{paramas.saliencyadd}/media{media}.jpg')
    G=saliencyadd[:,:,0]*0.5
    G=G.astype(int)
    # print(G.shape)
    G=G.astype(np.uint8)
    # print(G)
    # print(G.dtype)
    for sample in samples:
        R=np.full((576,720), r,dtype=np.uint8)
        # print(R.dtype)
        img=cv2.imread(f'{paramas.gazepointmap}{sca}/media{media}/{sample}')
        # print(img)
        B=img[:,:,0]
        # B.astype(np.float64)
        # print(B.dtype)
        merged = cv2.merge([B,G,R])
        cv2.imwrite(f'{paramas.saliencyaddgazergb}{r}/media{media}/{sample}',merged)


np.set_printoptions(threshold=np.inf)
pa=paramas()
if not os.path.exists(pa.saliencyadd):
    os.makedirs(pa.saliencyadd)
process_list=[]

for i in range(1,15):  #开启5个子进程执行fun1函数
    # if i!=14:
    p = Process(target=gazeaddsaliencymm,args=(pa,15,i)) #实例化进程对象
    p.start()
    process_list.append(p)

for i in process_list:
    i.join()


'''
def gazeaddsaliency(paramas,sca,media):
    if not os.path.exists(f'{paramas.saliencyaddgaze}/media{media}'):
        os.makedirs(f'{paramas.saliencyaddgaze}/media{media}')
    samples=os.listdir(f'{paramas.gazepointmap}{sca}/media{media}')
    saliencyadd=cv2.imread(f'{paramas.saliencyadd}/media{media}.jpg')
    for sample in samples:
        img=cv2.imread(f'{paramas.gazepointmap}{sca}/media{media}/{sample}')
        img1=cv2.addWeighted(saliencyadd,0.4,img,0.6,0)
        cv2.imwrite(f'{paramas.saliencyaddgaze}/media{media}/{sample}',img1)
'''