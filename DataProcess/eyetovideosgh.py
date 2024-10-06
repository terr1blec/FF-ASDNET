import os
from pathlib import Path
import cv2
import numpy.matlib
import numpy as np
from PIL import Image
import pandas as pd
from multiprocessing import Process

work_dir = Path.cwd()
# 加载数据

def chazhi(pathdata,pathvideo,savepath,media):
    # print("x")
    #pathdata是读取数据的路径，media1的上级路径，pathvideo是视频路径，视频名字是media1的形式，savepath是插值后文件保存的路径，media是媒体名，如media1
    if not os.path.exists(f'{savepath}area/{media}'):
        os.makedirs(f'{savepath}area/{media}')

    ###huoqu frame_num
    videoinpath = f'{pathvideo}/{media}.avi'

    capture = cv2.VideoCapture(videoinpath)

    if capture.isOpened():
        time_frames = 0
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        print('视频打开失败！')
    ######

    files=os.listdir(f'{pathdata}/{media}')
    for file in files:
        data = pd.read_excel(f'{pathdata}/{media}/{file}',header=None)

        data[1].fillna(0,inplace=True)   # 空值补0
        data[2].fillna(0,inplace=True)   # 空值补0
        # print("data:\n", data)
        dataNp= data.drop(columns=0, inplace=False).values.astype('float32')


        resized = cv2.resize(dataNp, (2,int(frame_num)),interpolation =cv2.INTER_AREA) ##高帧眼动数据差值到视频帧数
        #final_data = np.transpose(resized)
        ###
        t= resized.shape[0]
        a =np.linspace(1,t,t)
        final_data=np.column_stack((np.transpose(a),resized))     #对插值后的眼动数据加上序号


        data_df = pd.DataFrame(final_data)
        writer = pd.ExcelWriter(f'{savepath}area/{media}/{file}')
        data_df.to_excel(writer,index=False,float_format='%.5f',header=None)
        writer.save()










def makevideo(datapath,videopath,savevideopath,savevideoframes,media):
    # print("**********************************************************88")
    videoinpath  = f'{videopath}/{media}.avi'

    if not os.path.exists(f'{savevideopath}area/{media}'):
        os.makedirs(f'{savevideopath}area/{media}')

    files=os.listdir(f'{datapath}area/{media}')
    for file in files:
        videooutpath = f'{savevideopath}area/{media}/{file[:-5]}.avi'
        capture     = cv2.VideoCapture(videoinpath  )
        fourcc      = cv2.VideoWriter_fourcc(*'MJPG')
        writer      = cv2.VideoWriter(videooutpath ,fourcc, 25.0, (720,576), True)
        print("1")
        if capture.isOpened():
            time_frames = 0
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            while True:
                ret,img_src = capture.read()
                if not ret:
                    break
                img_out = op_one_img(img_src,frame_num,time_frames,datapath,file,savevideoframes,media)    # 自己写函数op_one_img()逐帧处理
                writer.write(img_out)
                time_frames = time_frames+1
        else:
            print('视频打开失败！')
        writer.release()


def op_one_img(frame,frame_num,time_frames,datapath,file,savevideoframes,media):
    if not os.path.exists(f'{savevideoframes}/{media}/{file[:-5]}'):
        os.makedirs(f'{savevideoframes}/{media}/{file[:-5]}')
    data=pd.read_excel(f'{datapath}area/{media}/{file}',header=None)
    width,height, chanel= frame.shape
    if data[1][time_frames] == 0:
        return frame
    else:
        width_lignt ,height_light  =int(data[1][time_frames]),int(data[2][time_frames])
        att_map = gauss(width,height,width_lignt ,height_light)
        frame = Image.fromarray(frame)
        image = Image.blend(frame,att_map,0.5)

        # f = image  # 获取当前图像
        # f.save(f'{savevideoframes}/{media}/{file[:-5]}/{time_frames}.png')
        #plt.figure()
        #plt.imshow(image)
        #plt.show()
        image= np.array(image)
        return image

def gauss(IMAGE_HEIGHT,IMAGE_WIDTH,center_x,center_y):      #provide gauss RGB map


    R = np.sqrt(center_x**2 + center_y**2)

    Gauss_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

    # 利用 for 循环 实现
    '''    for i in range(IMAGE_HEIGHT):

        for j in range(IMAGE_WIDTH):

            dis = np.sqrt((i-center_y)**2+(j-center_x)**2)

            Gauss_map[i, j] = np.exp(-0.5*dis/R)
    '''
    # 直接利用矩阵运算实现

    mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)

    mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)

    x1 = np.arange(IMAGE_WIDTH)

    x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)

    y1 = np.arange(IMAGE_HEIGHT)

    y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)

    y_map = np.transpose(y_map)

    Gauss_map = np.sqrt((x_map-mask_x)**2+(y_map-mask_y)**2)

    Gauss_map = np.exp(-0.5*Gauss_map/R)        #shengcheng mengban gray map

    #mengban tu bian cai tu
    norm_img = np.zeros(Gauss_map.shape)
    cv2.normalize(Gauss_map, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_RAINBOW)  # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
    # 显示和保存生成的图像

    #plt.figure()

    #plt.imshow(heat_img)

    #plt.imsave('out_2.jpg', heat_img, cmap=plt.cm.gray)

    #plt.show()
    heat_img = Image.fromarray(heat_img)
    return heat_img


def medias(datapath,videopath,savedatapath,savevideopath,savevideoframe,i):
    media = 'media' + str(i)
    chazhi(datapath, videopath, savedatapath, media)
    print("**********")
    makevideo(savedatapath, videopath, savevideopath, savevideoframe,media)



datapath=("./gazepointimg/indexxy")
savedatapath='./gazepointimg/indexxy_save'
videopath='./video'
savevideopath='./gazepointvideo/attentionvideo'
savevideoframepath='./gazepointvideo/attentionvideoframe'

process_list = []
for i in range(1,15):  #开启5个子进程执行fun1函数
     p = Process(target=medias,args=(datapath,videopath,savedatapath,savevideopath,savevideoframepath,i)) #实例化进程对象
     p.start()
     process_list.append(p)

for i in process_list:
    i.join()
