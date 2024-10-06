import os
import shutil
from multiprocessing import Process

def videos2class(ASDpath,TDpath,videopath,media):
    files=os.listdir(f'{videopath}/media{media}')
    for file in files:
        if file[0]=='n':
            shutil.copy(f'{videopath}/media{media}/{file}', f'{TDpath}/media{media}_{file}')
        else:
            shutil.copy(f'{videopath}/media{media}/{file}', f'{ASDpath}/media{media}_{file}')


videopath='gazepointvideo/heatmapvideo'
ASDpath='ASD'
TDpath='TD'
if not os.path.exists(ASDpath):
    os.makedirs(ASDpath)
if not os.path.exists(TDpath):
    os.makedirs(TDpath)

process_list = []
for i in range(1,15):  #开启5个子进程执行fun1函数
     p = Process(target=videos2class,args=(ASDpath,TDpath,videopath,i)) #实例化进程对象
     p.start()
     process_list.append(p)

for i in process_list:
    i.join()
