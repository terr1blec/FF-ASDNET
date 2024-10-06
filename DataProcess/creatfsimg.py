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
绘制时序视觉信息特征图像
gazepointfsjoint1d(paramas,size,radius,media)
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
        self.gazepointfsjoint1d='gazepointimg/fsjoint1d123'
        self.newgazepointfsjoint1d = 'newgazepointimg/fsjoint1d'
        self.fixationfsjoint1d='fixationimg/fsjoint1d'
        self.newfixationfsjoint1d='newfixationimg/fsjoint1d'
        self.saccadefixationfsjoint1d='saccadefixationimg/fsjoint1d'
        self.newsaccadefixationfsjoint1d='newsaccadefixationimg/fsjoint1d'
        self.videosaliencyhighlight='gazepointimg/9videosaliencyhighlight'
        self.sequence='gazepointimg/sequence'
        self.processvideosaliencyhighlight='gazepointimg/p9videosaliencyhighlight80'
        self.cutgazepointfsjoint1d='gazepointimg/cutfsjoint1d'
        self.cutlapgazepointfsjoint1d = 'gazepointimg/cutlapfsjoint1d'


def gazepointfsjoint1d(paramas,size,radius,media):
    if not os.path.exists(os.path.join(paramas.gazepointfsjoint1d+str(radius), media)):
        os.makedirs(os.path.join(paramas.gazepointfsjoint1d+str(radius), media))
    saliencys=os.listdir(paramas.saliency+'/'+media)
    saliencys.sort(key=lambda x: int(x[5:-8]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.gazepointimgcoor+'/'+media)
    d=radius  #半径50
    for sample in coors:
        sheet=pd.read_excel(paramas.gazepointimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        print("##################"+str(sheet))
        channel1=np.ones((50, int(50*size[media]), 1))
        #print("##################"+str(channel1))
        channel2=channel1+channel1
        channel3=channel2+channel1
        jointimg0=np.concatenate((channel1,channel2,channel3),axis=2)
        # jointimg0=np.zeros((50, int(50*size[media]), 3))
        for i in range(size[media]):
            for co in sheet:
                if co[0] == i + 1:
                    img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]))
                    #print("##################")
                    if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                        starty = round(float(co[1]))
                    elif round(float(co[1])) < d:
                        starty = d
                    else:
                        starty = 720 - d
                    if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                        startx = round(float(co[2]))
                    elif round(float(co[2])) < d:
                        startx = d
                    else:
                        startx = 576 - d
                    img = img[startx - d:startx + d, starty - d:starty + d]
                    img = cv2.resize(img, (50, 50))
                    jointimg0[ 0: 50,i * 50:i * 50 + 50] = img
        cv2.imwrite(paramas.gazepointfsjoint1d+str(radius)+'/' + media +'/' + sample.split('.')[0]  +'.jpg', jointimg0)


pa=paramas()

process_list3=[]
for i in range(1,15):  #开启5个子进程执行fun1函数
    p=Process(target= gazepointfsjoint1d,args=(pa,size,50,'media'+str(i)))
    p.start()
    process_list3.append(p)
for i in process_list3:
    i.join()

# for i in range(2,5):  #开启5个子进程执行fun1函数
#     if i!=3  and i!=6 and i!=12:
#         videosaliencyhighlight(pa,size,80,'media'+str(i))


'''
def processvideo(paramas,size,sca,media):
    if not os.path.exists(f'{paramas.processvideosaliencyhighlight}/{media}'):
        os.makedirs(f'{paramas.processvideosaliencyhighlight}/{media}')
    files=os.listdir(f'{paramas.videosaliencyhighlight}{sca}/{media}')
    for file in files:
        if not os.path.exists(f'{paramas.processvideosaliencyhighlight}/{media}/{file}'):
            os.makedirs(f'{paramas.processvideosaliencyhighlight}/{media}/{file}')
        imgs=os.listdir(f'{paramas.videosaliencyhighlight}{sca}/{media}/{file}')
        for i in range(len(imgs)):
            img=Image.open(f'{paramas.videosaliencyhighlight}{sca}/{media}/{file}/{imgs[i]}')
            img=img.resize((864,690),resample=Image.BILINEAR)
            left = (864- 600) // 2
            top = (690 - 600) // 2
            right = (864 + 600) // 2
            bottom = (690 + 600) // 2

            # Crop the center of the image
            img = img.crop((left, top, right, bottom))
            # img=img.crop((600,600))
            img=img.resize((100,100),resample=Image.BILINEAR)
            img.save(f'{paramas.processvideosaliencyhighlight}/{media}/{file}/{imgs[i]}')


def videosaliencyhighlight(paramas,size,sca,media):
    if not os.path.exists(f'{paramas.videosaliencyhighlight}{sca}/{media}'):
        os.makedirs(f'{paramas.videosaliencyhighlight}{sca}/{media}')
    saliencys = os.listdir(paramas.saliency + '/' + media)
    saliencys.sort(key=lambda x: int(x[5:-8]))
    coors = os.listdir(paramas.gazepointimgcoor + '/' + media)
    d = 50  # 半径50
    for sample in coors:
        sheet=pd.read_excel(paramas.gazepointimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        if not os.path.exists(f'{paramas.videosaliencyhighlight}{sca}/{media}/{sample[:-4]}'):
            os.makedirs(f'{paramas.videosaliencyhighlight}{sca}/{media}/{sample[:-4]}')
        for i in range(size[media]):
            img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]),cv2.IMREAD_GRAYSCALE).astype(np.float32)
            startx, starty,bz = 0, 0,0
            for co in sheet:
                bz=0
                if co[0] == i + 1:
                    bz=1
                    if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                        starty = round(float(co[1]))
                    elif round(float(co[1])) < d:
                        starty = d
                    else:
                        starty = 720 - d
                    if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                        startx = round(float(co[2]))
                    elif round(float(co[2])) < d:
                        startx = d
                    else:
                        startx = 576 - d
                    img=hightlight(img,startx,starty,d,sca)
                    break
            if bz==0:
                img1 = cv2.blur(img,(9,9))
                cv2.imwrite(f'{paramas.videosaliencyhighlight}{sca}/{media}/{sample[:-4]}/{saliencys[i][:-4]}.jpg', img1)
            else:
                img=mask(img,startx,starty,d)
                cv2.imwrite(f'{paramas.videosaliencyhighlight}{sca}/{media}/{sample[:-4]}/{saliencys[i][:-4]}.jpg',img)

# @jit(nopython=True)
def hightlight(frame,startx,starty,d,sca):
    img=frame

    mstartx = startx - 2 * d
    if mstartx <= 0:
        mstartx = 0
    mendx = startx + 2 * d
    if mendx > 576:
        mendx = 576
    mstarty = starty - 2 * d
    if mstarty <= 0:
        mstarty = 0
    mendy = starty + 2 * d
    if mendy > 720:
        mendy = 720
    xs=[]
    xs.append(0.0)
    xd=[[0]]
    yd = [[0]]
    x = [[0]]
    y = [[0]]
    for i in range(startx-d,startx+d):
        for j in range(starty - d, starty + d):
            bz=0
            for z in range(len(xs)):
                if img[i][j]==xs[z] or img[i][j] == 0:
                    bz=1
                    break
            if bz==0:
                xs.append(img[i][j])

    for k in range(1,len(xs)) :
        xin,yin,xout,yout=[0],[0],[0],[0]
        for i in range(mstartx,mendx):
            for j in range(mstarty,mendy):
                if img[i][j]==xs[k]:
                    if i<=startx+d and i>=startx-d and j<=starty+d and j>=starty-d:
                        xin.append(i)
                        yin.append(j)
                    else:
                        xout.append(i)
                        yout.append(j)
        xd.append(xin)
        yd.append(yin)
        x.append(xout)
        y.append(yout)

    for i in range(1,len(xd)):
        if len(xd[i])>len(x[i]):
            for j in range(len(xd[i])):
                img[xd[i][j]][yd[i][j]]=img[xd[i][j]][yd[i][j]]+sca
            for j in range(len(x[i])):
                img[x[i][j]][y[i][j]]=img[x[i][j]][y[i][j]]+sca

    return frame

@jit(nopython=True)
def mask(frame,startx,starty,d):
    img=frame
    for i in range(len(frame)):
        for j in range(len(frame[i])):
            if pow(i-startx,2)+pow(j-starty,2)>pow(d,2) and pow(i-startx,2)+pow(j-starty,2)<=pow(2*d,2):
                bjsx=i-2
                bjex=i+2
                bjsy=j-2
                bjey=j+2
                if i < 2:
                    bjsx = 0
                if i > 573:
                    bjex = 575
                if j < 2:
                    bjsy = 0
                if j > 717:
                    bjey = 719
                img[i][j]=np.sum(frame[bjsx:bjex+1,bjsy:bjey+1])/25
            elif pow(i-startx,2)+pow(j-starty,2)>pow(2*d,2):
                bjsx = i - 4
                bjex = i + 4
                bjsy = j - 4
                bjey = j + 4
                if i <4:
                    bjsx = 0
                if i >571:
                    bjex = 575
                if j <4:
                    bjsy = 0
                if j > 715:
                    bjey = 719
                img[i][j] = np.sum(frame[bjsx:bjex + 1, bjsy:bjey + 1]) / 81

    return img

def copygazepointfsjoint1d(paramas,size,radius,media):
    if not os.path.exists(os.path.join(paramas.gazepointfsjoint1d+str(radius), media)):
        os.makedirs(os.path.join(paramas.gazepointfsjoint1d+str(radius), media))
    saliencys=os.listdir(paramas.saliency+'/'+media)
    saliencys.sort(key=lambda x: int(x[5:-8]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.gazepointimgcoor+'/'+media)
    d=radius  #半径50
    if media != 'media2':
        if size['media2']%size[media]!=0:
            copynum = size['media2']//size[media]+1
        else:
            copynum = size['media2'] // size[media]
    for sample in coors:
        sheet=pd.read_excel(paramas.gazepointimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        channel1=np.ones((50, int(50*size[media]), 1))
        channel2=channel1+channel1
        channel3=channel2+channel1
        jointimg0=np.concatenate((channel1,channel2,channel3),axis=2)
        # jointimg0=np.zeros((50, int(50*size[media]), 3))
        for i in range(size[media]):
            for co in sheet:
                if co[0] == i + 1:
                    img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]))
                    if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                        starty = round(float(co[1]))
                    elif round(float(co[1])) < d:
                        starty = d
                    else:
                        starty = 720 - d
                    if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                        startx = round(float(co[2]))
                    elif round(float(co[2])) < d:
                        startx = d
                    else:
                        startx = 576 - d
                    img = img[startx - d:startx + d, starty - d:starty + d]
                    img = cv2.resize(img, (50, 50))
                    jointimg0[ 0: 50,i * 50:i * 50 + 50] = img
        jointimg=jointimg0
        if media != 'media2':
            for i in range(copynum):
                jointimg=np.concatenate((jointimg,jointimg0),axis=1)
            jointimg=jointimg[0:50,0:14300]
        cv2.imwrite(paramas.gazepointfsjoint1d+str(radius)+'/' + media +'/' + sample.split('.')[0]  +'.jpg', jointimg)

def cutgazepointfsjoint1d(paramas,size,radius,media):
    if not os.path.exists(os.path.join(paramas.cutgazepointfsjoint1d, media)):
        os.makedirs(os.path.join(paramas.cutgazepointfsjoint1d, media))
    saliencys=os.listdir(paramas.saliency+'/'+media)
    saliencys.sort(key=lambda x: int(x[5:-8]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.gazepointimgcoor+'/'+media)
    d=radius  #半径50
    zs=size[media]//50
    ys=size[media]%50
    if ys > 25:
        zs+=1
    for sample in coors:
        sheet=pd.read_excel(paramas.gazepointimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        for k in range(zs):
            e=0
            if k*50+50>=size[media]:
                mw=size[media]
                jointimg0 = np.zeros((50, int(50 * (size[media]-k*50)), 3))
            else:
                mw=(k+1)*50
                jointimg0 = np.zeros((50, int(50 * 50), 3))
            for i in range(k*50,mw):
                for co in sheet:
                    if co[0] == i + 1:
                        e=1
                        img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]))
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        jointimg0[ 0: 50,(i-k*50) * 50:(i-k*50) * 50 + 50] = img
            if e==1:
                cv2.imwrite(paramas.cutgazepointfsjoint1d+'/' + media +'/' + sample.split('.')[0]+'_'+str(k+1)  +'.jpg', jointimg0)

def cutlapgazepointfsjoint1d(paramas,size,radius,media):
    if not os.path.exists(os.path.join(paramas.cutlapgazepointfsjoint1d, media)):
        os.makedirs(os.path.join(paramas.cutlapgazepointfsjoint1d, media))
    saliencys=os.listdir(paramas.saliency+'/'+media)
    saliencys.sort(key=lambda x: int(x[5:-8]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.gazepointimgcoor+'/'+media)
    d=radius  #半径50
    zs=(size[media]-50)//30+1
    ys=(size[media]-50)%30
    if ys > 15:
        zs+=1
    for sample in coors:
        sheet=pd.read_excel(paramas.gazepointimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        for k in range(zs):
            e=0
            jointimg0 = np.zeros((50, int(50 * 50), 3))
            if zs==1:
                mw=50
                ks=0
            if k==zs-1:
                mw=size[media]
                ks=size[media]-50
            else:
                ks=k*30
                mw=ks+50
            for i in range(ks,mw):
                for co in sheet:
                    if co[0] == i + 1:
                        e=1
                        img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]))
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        jointimg0[ 0: 50,(i-ks) * 50:(i-ks) * 50 + 50] = img
            cv2.imwrite(paramas.cutlapgazepointfsjoint1d+'/' + media +'/' + sample.split('.')[0]+'_'+str(k+1)  +'.jpg', jointimg0)

#fixation也可
def newgazepointfsjoint1d(paramas,size,media):
    if not os.path.exists(os.path.join(paramas.newgazepointfsjoint1d, media)):
        os.makedirs(os.path.join(paramas.newgazepointfsjoint1d, media))
    saliencys=os.listdir(paramas.newsaliency+'/'+media)
    saliencys.sort(key=lambda x: int(x[:-4]))
    frames = os.listdir(paramas.frame + '/' + media)
    frames.sort(key=lambda x: int(x[5:-4]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.gazepointimgcoor+'/'+media)
    d=50  #半径50
    for sample in coors:
        sheet=pd.read_excel(paramas.gazepointimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        jointimg0=np.zeros((50, int(50*size[media]), 3))
        for i in range(size[media]):
            for co in sheet:
                if co[0] == i + 1:
                    img = cv2.imread(os.path.join(paramas.newsaliency, media, saliencys[i]))
                    img = cv2.resize(img, (720, 576))
                    frame = cv2.imread(os.path.join(paramas.frame, media, frames[i]))
                    img = frameimg(frame, img)
                    if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                        starty = round(float(co[1]))
                    elif round(float(co[1])) < d:
                        starty = d
                    else:
                        starty = 720 - d
                    if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                        startx = round(float(co[2]))
                    elif round(float(co[2])) < d:
                        startx = d
                    else:
                        startx = 576 - d
                    img = img[startx - d:startx + d, starty - d:starty + d]
                    img = cv2.resize(img, (50, 50))
                    jointimg0[ 0: 50,i * 50:i * 50 + 50] = img
        cv2.imwrite(paramas.newgazepointfsjoint1d +'/' + media +'/' + sample.split('.')[0]  +'.jpg', jointimg0)

def saccadefixationfsjoint1d(paramas,size,sca,media):
    if not os.path.exists(os.path.join(paramas.saccadefixationfsjoint1d+str(sca), media)):
        os.makedirs(os.path.join(paramas.saccadefixationfsjoint1d+str(sca), media))
    saliencys=os.listdir(paramas.saliency+'/'+media)
    saliencys.sort(key=lambda x: int(x[5:-8]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.saccadefixationimgcoor+'/'+media)
    d=50  #半径50
    for sample in coors:
        sheet=pd.read_excel(paramas.saccadefixationimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        jointimg0=np.zeros((50, int(50*size[media]), 3))
        for i in range(size[media]):
            for co in sheet:
                if co[0] == i + 1:
                    img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]))
                    if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                        starty = round(float(co[1]))
                    elif round(float(co[1])) < d:
                        starty = d
                    else:
                        starty = 720 - d
                    if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                        startx = round(float(co[2]))
                    elif round(float(co[2])) < d:
                        startx = d
                    else:
                        startx = 576 - d
                    img = img[startx - d:startx + d, starty - d:starty + d]
                    img = cv2.resize(img, (50, 50))
                    if co[4] == 'Saccade':
                        img = scam(img, sca)
                    jointimg0[ 0: 50,i * 50:i * 50 + 50] = img
        cv2.imwrite(paramas.saccadefixationfsjoint1d+str(sca) +'/' + media +'/' + sample.split('.')[0]  +'.jpg', jointimg0)

def newsaccadefixationfsjoint1d(paramas,size,sca,media):
    if not os.path.exists(os.path.join(paramas.newsaccadefixationfsjoint1d+str(sca), media)):
        os.makedirs(os.path.join(paramas.newsaccadefixationfsjoint1d+str(sca), media))
    saliencys=os.listdir(paramas.newsaliency+'/'+media)
    # saliencys.sort(key=lambda x: int(x[5:-8]))
    saliencys.sort(key=lambda x:int(x[0:-4]))
    frames = os.listdir(paramas.frame + '/' + media)
    frames.sort(key=lambda x: int(x[5:-4]))
    coors=os.listdir(paramas.saccadefixationimgcoor+'/'+media)
    d=50  #半径50
    for sample in coors:
        sheet=pd.read_excel(paramas.saccadefixationimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        jointimg0=np.zeros((50, int(50*size[media]), 3))
        for i in range(size[media]):
            for co in sheet:
                if co[0] == i + 1:
                    img = cv2.imread(os.path.join(paramas.newsaliency, media, saliencys[i]))
                    img = cv2.resize(img, (720, 576))
                    frame = cv2.imread(os.path.join(paramas.frame, media, frames[i]))
                    img = frameimg(frame, img)
                    if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                        starty = round(float(co[1]))
                    elif round(float(co[1])) < d:
                        starty = d
                    else:
                        starty = 720 - d
                    if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                        startx = round(float(co[2]))
                    elif round(float(co[2])) < d:
                        startx = d
                    else:
                        startx = 576 - d
                    img = img[startx - d:startx + d, starty - d:starty + d]
                    img = cv2.resize(img, (50, 50))
                    if co[4] == 'Saccade':
                        img = scam(img, sca)
                    jointimg0[ 0: 50,i * 50:i * 50 + 50] = img
        cv2.imwrite(paramas.newsaccadefixationfsjoint1d+str(sca) +'/' + media +'/' + sample.split('.')[0]  +'.jpg', jointimg0)

def fsjoint(paramas,size,gap,sca,media):#sca指Saccade图像块的透明度
    if not os.path.exists(os.path.join(paramas.saveimgjoint+str(sca), media)):
        os.makedirs(os.path.join(paramas.saveimgjoint+str(sca), media))
    saliencys=os.listdir(paramas.saimgpath+'/'+media)
    saliencys.sort(key=lambda x: int(x[5:-8]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.pathcoor+'/'+media)
    h,w=imgsize(size,media,gap)
    col=w
    h=int(h*50)
    w=int(w*50)
    d=50  #半径50
    for sample in coors:
        sheet=pd.read_excel(paramas.pathcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        jointimg0=np.zeros((h, w, 3))
        for i in range(int(gap*25),len(saliencys)):
            bx=[]
            for co in sheet:
                if co[0]==i+1:
                    if co[4]=='Saccade':
                        bx.append(co)
                    else:
                        bx.append(co)
                        break
            if len(bx)!=0:
                img=cv2.imread(os.path.join(paramas.saimgpath,media,saliencys[i]))
                if round(float(bx[-1][1]))<=720-d and round(float(bx[-1][1]))>=d:
                    starty=round(float(bx[-1][1]))
                elif round(float(bx[-1][1]))<d:
                    starty=d
                else:
                    starty=720-d
                if round(float(bx[-1][2]))<=576-d and round(float(bx[-1][2]))>=d:
                    startx=round(float(bx[-1][2]))
                elif round(float(bx[-1][2]))<d:
                    startx=d
                else:
                    startx=576-d
                img= img[startx - d:startx + d, starty - d:starty + d]
                img=cv2.resize(img,(50,50))
                if bx[-1][4]=='Saccade':
                    for o in range(len(img)):
                        for p in range(len(img[o])):
                            for q in range(len(img[o][p])):
                                img[o][p][q]=round(sca*img[o][p][q])
                a=(i)//col
                b=(i)%col
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50]=img

        cv2.imwrite(paramas.saveimgjoint +str(sca)+'/' + media +'/' + sample.split('.')[0]  +'.jpg', jointimg0)

def fixationfsjointss(paramas,size,gap,media):
    if not os.path.exists(os.path.join(paramas.fixationfsjoint, media)):
        os.makedirs(os.path.join(paramas.fixationfsjoint, media))
    saliencys=os.listdir(paramas.saliency+'/'+media)
    saliencys.sort(key=lambda x: int(x[5:-8]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.fixationimgcoor+'/'+media)
    h,w=imgsize(size,'media3',gap)
    hmw=int(h*w)
    qz=size[media]//hmw
    ys=size[media]%hmw
    col=w
    h=int(h*50)
    w=int(w*50)
    d=50  #半径50
    for sample in coors:
        sheet=pd.read_excel(paramas.fixationimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        jointimg0=np.zeros((h, w, 3))
        for j in range(ys):
            fpsf=0
            finalimg=np.zeros((50,50,3))
            for i in range(int(j*(qz+1)),int((j+1)*(qz+1))):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf+=1
                        img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]))
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg


        for j in range(ys,hmw):
            fpsf = 0
            finalimg = np.zeros((50, 50, 3))
            for i in range(int(j*qz),int((j+1)*qz)):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf += 1
                        img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]))
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg
        cv2.imwrite(paramas.fixationfsjoint +'/' + media +'/' + sample.split('.')[0]  +'.jpg', jointimg0)

def saccadefixationfsjointss(paramas,size,gap,sca,media):
    if not os.path.exists(os.path.join(paramas.saccadefixationfsjoint+str(sca), media)):
        os.makedirs(os.path.join(paramas.saccadefixationfsjoint+str(sca), media))
    saliencys=os.listdir(paramas.saliency+'/'+media)
    saliencys.sort(key=lambda x: int(x[5:-8]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.saccadefixationimgcoor+'/'+media)
    h,w=imgsize(size,'media13',gap)
    hmw=int(h*w)
    qz=size[media]//hmw
    ys=size[media]%hmw
    col=w
    h=int(h*50)
    w=int(w*50)
    d=50  #半径50
    for sample in coors:
        sheet=pd.read_excel(paramas.saccadefixationimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        jointimg0=np.zeros((h, w, 3))
        for j in range(ys):
            fpsf=0
            finalimg=np.zeros((50,50,3))
            for i in range(int(j*(qz+1)),int((j+1)*(qz+1))):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf+=1
                        img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]))
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        if co[4] == 'Saccade':
                            img=scam(img,sca)
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg


        for j in range(ys,hmw):
            fpsf = 0
            finalimg = np.zeros((50, 50, 3))
            for i in range(int(j*qz),int((j+1)*qz)):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf += 1
                        img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]))
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        if co[4] == 'Saccade':
                            img=scam(img,sca)
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg
        cv2.imwrite(paramas.saccadefixationfsjoint +str(sca)+'/' + media +'/' + sample.split('.')[0]  +'.jpg', jointimg0)

#使用新方法生成的显著性图像进行拼接
def newsaccadefixationfsjointss(paramas,size,gap,sca,media):
    if not os.path.exists(os.path.join(paramas.newsaccadefixationfsjoint+str(sca), media)):
        os.makedirs(os.path.join(paramas.newsaccadefixationfsjoint+str(sca), media))
    saliencys=os.listdir(paramas.newsaliency+'/'+media)
    saliencys.sort(key=lambda x: int(x[:-4]))
    frames = os.listdir(paramas.frame + '/' + media)
    frames.sort(key=lambda x: int(x[5:-4]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.saccadefixationimgcoor+'/'+media)
    h,w=imgsize(size,'media13',gap)
    hmw=int(h*w)
    qz=size[media]//hmw
    ys=size[media]%hmw
    col=w
    h=int(h*50)
    w=int(w*50)
    d=50  #半径50
    for sample in coors:
        sheet=pd.read_excel(paramas.saccadefixationimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        jointimg0=np.zeros((h, w, 3))
        for j in range(ys):
            fpsf=0
            finalimg=np.zeros((50,50,3))
            for i in range(int(j*(qz+1)),int((j+1)*(qz+1))):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf+=1
                        img = cv2.imread(os.path.join(paramas.newsaliency, media, saliencys[i]))
                        img=cv2.resize(img,(720,576))
                        frame = cv2.imread(os.path.join(paramas.frame, media, frames[i]))
                        img = frameimg(frame, img)
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        if co[4] == 'Saccade':
                            img=scam(img,sca)
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg


        for j in range(ys,hmw):
            fpsf = 0
            finalimg = np.zeros((50, 50, 3))
            for i in range(int(j*qz),int((j+1)*qz)):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf += 1
                        img = cv2.imread(os.path.join(paramas.newsaliency, media, saliencys[i]))
                        img=cv2.resize(img,(720,576))
                        frame = cv2.imread(os.path.join(paramas.frame, media, frames[i]))
                        img = frameimg(frame, img)
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        if co[4] == 'Saccade':
                            img=scam(img,sca)
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg
        cv2.imwrite(paramas.newsaccadefixationfsjoint +str(sca)+'/' + media +'/' + sample.split('.')[0]  +'.jpg', jointimg0)

def newfixationfsjointss(paramas,size,gap,media):
    if not os.path.exists(os.path.join(paramas.newfixationfsjoint, media)):
        os.makedirs(os.path.join(paramas.newfixationfsjoint, media))
    saliencys=os.listdir(paramas.newsaliency+'/'+media)
    saliencys.sort(key=lambda x: int(x[:-4]))
    frames = os.listdir(paramas.frame + '/' + media)
    frames.sort(key=lambda x: int(x[5:-4]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.fixationimgcoor+'/'+media)
    h,w=imgsize(size,'media13',gap)
    hmw=int(h*w)
    qz=size[media]//hmw
    ys=size[media]%hmw
    col=w
    h=int(h*50)
    w=int(w*50)
    d=50  #半径50
    for sample in coors:
        sheet=pd.read_excel(paramas.fixationimgcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        jointimg0=np.zeros((h, w, 3))
        for j in range(ys):
            fpsf=0
            finalimg=np.zeros((50,50,3))
            for i in range(int(j*(qz+1)),int((j+1)*(qz+1))):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf+=1
                        img = cv2.imread(os.path.join(paramas.newsaliency, media, saliencys[i]))
                        img=cv2.resize(img,(720,576))
                        frame = cv2.imread(os.path.join(paramas.frame, media, frames[i]))
                        img = frameimg(frame, img)
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg


        for j in range(ys,hmw):
            fpsf = 0
            finalimg = np.zeros((50, 50, 3))
            for i in range(int(j*qz),int((j+1)*qz)):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf += 1
                        img = cv2.imread(os.path.join(paramas.newsaliency, media, saliencys[i]))
                        img=cv2.resize(img,(720,576))
                        frame = cv2.imread(os.path.join(paramas.frame, media, frames[i]))
                        img = frameimg(frame, img)
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg
        cv2.imwrite(paramas.newfixationfsjoint +'/' + media +'/' + sample.split('.')[0]  +'.jpg', jointimg0)

def newgazepointfsjointss(paramas, size, gap, media):
    if not os.path.exists(os.path.join(paramas.newgazepointfsjoint, media)):
        os.makedirs(os.path.join(paramas.newgazepointfsjoint, media))
    saliencys = os.listdir(paramas.newsaliency + '/' + media)
    saliencys.sort(key=lambda x: int(x[:-4]))
    frames = os.listdir(paramas.frame + '/' + media)
    frames.sort(key=lambda x: int(x[5:-4]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors = os.listdir(paramas.gazepointimgcoor + '/' + media)
    h, w = imgsize(size, 'media13', gap)
    # h,w=6,6
    hmw = int(h * w)
    qz = size[media] // hmw
    ys = size[media] % hmw
    col = w
    h = int(h * 50)
    w = int(w * 50)
    d = 50  # 半径50
    for sample in coors:
        sheet = pd.read_excel(paramas.gazepointimgcoor + '/' + media + '/' + sample)
        sheet = sheet.values.tolist()
        jointimg0 = np.zeros((h, w, 3))
        for j in range(ys):
            fpsf = 0
            finalimg = np.zeros((50, 50, 3))
            for i in range(int(j * (qz + 1)), int((j + 1) * (qz + 1))):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf += 1
                        img = cv2.imread(os.path.join(paramas.newsaliency, media, saliencys[i]))
                        img = cv2.resize(img, (720, 576))
                        frame = cv2.imread(os.path.join(paramas.frame, media, frames[i]))
                        img = frameimg(frame, img)
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg

        for j in range(ys, hmw):
            fpsf = 0
            finalimg = np.zeros((50, 50, 3))
            for i in range(int(j * qz), int((j + 1) * qz)):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf += 1
                        img = cv2.imread(os.path.join(paramas.newsaliency, media, saliencys[i]))
                        img = cv2.resize(img, (720, 576))
                        frame = cv2.imread(os.path.join(paramas.frame, media, frames[i]))
                        img = frameimg(frame, img)
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg
        cv2.imwrite(paramas.newgazepointfsjoint + '/' + media + '/' + sample.split('.')[0] + '.jpg', jointimg0)

@jit(nopython=True)
def frameimg(frame,img):
    for g in range(len(frame)):
        for e in range(len(frame[g])):
            for f in range(len(frame[g][e])):
                img[g][e][f] = round(frame[g][e][f] * img[g][e][0] / 255)
    return img

@jit(nopython=True)
def scam(img,sca):
    for o in range(len(img)):
        for p in range(len(img[o])):
            for q in range(len(img[o][p])):
                img[o][p][q] = round(sca * img[o][p][q])
    return img


def imgsize(size,media,gap):
    fps=size[media]-gap*25
    a=pow(fps,0.5)
    b=int(a)
    if pow(b,2)==fps:
        w=b
        h=b
    else:
        c=round(a)
        w1 = []
        h1 = []
        f1=[]
        e = c
        if c==b:
            d=c+1
        else:
            d=c
        h1.append(e)
        w1.append(d)
        f=e*d-fps
        f1.append(f)
        while f>=0:
            e-=1
            d+=1
            h1.append(e)
            w1.append(d)
            f = e * d-fps
            f1.append(f)
        f1.pop()
        ind=f1.index(min(f1))
        w=w1[ind]
        h=h1[ind]
    return h,w

def sequencess(paramas,size,media):
    if not os.path.exists(os.path.join(paramas.sequence, media)):
        os.makedirs(os.path.join(paramas.sequence, media))
    saliencys = os.listdir(paramas.saliency + '/' + media)
    saliencys.sort(key=lambda x: int(x[5:-8]))
    coors = os.listdir(paramas.gazepointimgcoor + '/' + media)
    d = 50  # 半径50
    for sample in coors:
        sheet = pd.read_excel(paramas.gazepointimgcoor + '/' + media + '/' + sample)
        sheet = sheet.values.tolist()
        if not os.path.exists(f'{paramas.sequence}{media}/{sample[:-4]}'):
            os.makedirs(f'{paramas.sequence}/{media}/{sample[:-4]}')
        for i in range(size[media]):
            startx, starty, bz = 0, 0, 0
            for co in sheet:
                bz = 0
                if co[0] == i + 1:
                    img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]), cv2.IMREAD_GRAYSCALE).astype(
                        np.float32)
                    bz = 1
                    if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                        starty = round(float(co[1]))
                    elif round(float(co[1])) < d:
                        starty = d
                    else:
                        starty = 720 - d
                    if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                        startx = round(float(co[2]))
                    elif round(float(co[2])) < d:
                        startx = d
                    else:
                        startx = 576 - d
                    img = img[startx - d:startx + d, starty - d:starty + d]
                    break
            if bz == 0:
                img=np.zeros((100,100,1))
            cv2.imwrite(f'{paramas.sequence}/{media}/{sample[:-4]}/{saliencys[i][:-4]}.jpg', img)

def gazepointfsjointss(paramas, size, gap, media):
    if not os.path.exists(os.path.join(paramas.gazepointfsjoint, media)):
        os.makedirs(os.path.join(paramas.gazepointfsjoint, media))
    saliencys = os.listdir(paramas.saliency + '/' + media)
    saliencys.sort(key=lambda x: int(x[5:-8]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors = os.listdir(paramas.gazepointimgcoor + '/' + media)
    h, w = imgsize(size, 'media3', gap)
    # h,w=6,6
    hmw = int(h * w)
    qz = size[media] // hmw
    ys = size[media] % hmw
    col = w
    h = int(h * 50)
    w = int(w * 50)
    d = 50  # 半径50
    for sample in coors:
        sheet = pd.read_excel(paramas.gazepointimgcoor + '/' + media + '/' + sample)
        sheet = sheet.values.tolist()
        jointimg0 = np.zeros((h, w, 3))
        for j in range(ys):
            fpsf = 0
            finalimg = np.zeros((50, 50, 3))
            for i in range(int(j * (qz + 1)), int((j + 1) * (qz + 1))):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf += 1
                        img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]))
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg

        for j in range(ys, hmw):
            fpsf = 0
            finalimg = np.zeros((50, 50, 3))
            for i in range(int(j * qz), int((j + 1) * qz)):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf += 1
                        img = cv2.imread(os.path.join(paramas.saliency, media, saliencys[i]))
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg
        cv2.imwrite(paramas.gazepointfsjoint + '/' + media + '/' + sample.split('.')[0] + '.jpg', jointimg0)

def allpointnewfsjointss(paramas,size,gap,media):
    if not os.path.exists(os.path.join(paramas.newsavejsss, media)):
        os.makedirs(os.path.join(paramas.newsavejsss, media))
    saliencys=os.listdir(paramas.newsaliency+'/'+media)
    saliencys.sort(key=lambda x: int(x[:-4]))
    frames = os.listdir(paramas.frame + '/' + media)
    frames.sort(key=lambda x: int(x[5:-4]))
    # saliencys.sort(key=lambda x:int(x[5:-4]))
    coors=os.listdir(paramas.pathcoor+'/'+media)
    # h,w=imgsize(size,'media3',gap)
    h,w=6,6
    hmw=int(h*w)
    qz=size[media]//hmw
    ys=size[media]%hmw
    col=w
    h=int(h*50)
    w=int(w*50)
    d=50  #半径50
    for sample in coors:
        sheet=pd.read_excel(paramas.pathcoor +'/' + media+'/' + sample)
        sheet=sheet.values.tolist()
        jointimg0=np.zeros((h, w, 3))
        for j in range(ys):
            fpsf=0
            finalimg=np.zeros((50,50,3))
            for i in range(int(j*(qz+1)),int((j+1)*(qz+1))):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf+=1
                        img = cv2.imread(os.path.join(paramas.newsaliency, media, saliencys[i]))
                        img=cv2.resize(img,(720,576))
                        frame = cv2.imread(os.path.join(paramas.frame, media, frames[i]))
                        img = frameimg(frame, img)
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg


        for j in range(ys,hmw):
            fpsf = 0
            finalimg = np.zeros((50, 50, 3))
            for i in range(int(j*qz),int((j+1)*qz)):
                for co in sheet:
                    if co[0] == i + 1:
                        fpsf += 1
                        img = cv2.imread(os.path.join(paramas.newsaliency, media, saliencys[i]))
                        img=cv2.resize(img,(720,576))
                        frame = cv2.imread(os.path.join(paramas.frame, media, frames[i]))
                        img = frameimg(frame, img)
                        if round(float(co[1])) <= 720 - d and round(float(co[1])) >= d:
                            starty = round(float(co[1]))
                        elif round(float(co[1])) < d:
                            starty = d
                        else:
                            starty = 720 - d
                        if round(float(co[2])) <= 576 - d and round(float(co[2])) >= d:
                            startx = round(float(co[2]))
                        elif round(float(co[2])) < d:
                            startx = d
                        else:
                            startx = 576 - d
                        img = img[startx - d:startx + d, starty - d:starty + d]
                        img = cv2.resize(img, (50, 50))
                        finalimg = finalimg + img
            if fpsf != 0:
                a = (j) // col
                b = (j) % col
                finalimg = finalimg / fpsf
                finalimg = np.around(finalimg)
                finalimg = finalimg.astype(int)
                jointimg0[a * 50:a * 50 + 50, b * 50:b * 50 + 50] = finalimg
        cv2.imwrite(paramas.newsavejsss +'/' + media +'/' + sample.split('.')[0]  +'.jpg', jointimg0)
'''