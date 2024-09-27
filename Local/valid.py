import os
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import xlwt
import torch
import random

perfix_name='10gzresnet'
print(perfix_name)
finalacc=0
for i in range(5):
    fo = open(f'v20220626/output/{perfix_name}_{i}/64_1e-06/test.txt', 'r')
    participants = []
    myFile = fo.read()
    myRecords = myFile.split('\n')
    for y in range(0, len(myRecords) - 1):
        participants.append(myRecords[y].split('\t'))  # 把表格数据存入
    label,pre=[],[]
    for j in range(len(participants)):
        label.append(int(participants[j][0]))
        pre.append(int(participants[j][1]))
#     acc=metrics.accuracy_score(label,pre)
#     print('round'+str(i)+': '+str(acc))
#     finalacc=finalacc+acc
# finalacc=finalacc/5
# print('mean: '+str(finalacc))
    cm = metrics.confusion_matrix(label,pre, labels=[0,1], sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    print('round'+str(i))
    print(cm)


fo = open(f'../../sample/participant.txt', 'r')
participants = []
myFile = fo.read()
myRecords = myFile.split('\n')
for y in range(0, len(myRecords) - 1):
    participants.append(myRecords[y].split('\t'))  # 把表格数据存入
for i in range(len(participants)):
    pzs = np.zeros(10)
    pcw = np.zeros(10)
    sezs = np.zeros(3)
    secw=np.zeros(3)
    print(i)
    sheet=xlwt.Workbook()
    sheet3 = sheet.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    f=open(f'../../sample/gazepoint/10/{i}/testsample.txt','r')
    line1=['participant','media1','media2','media4','media5','media7','media8','media9','media10','media11','media13']
    for j in range(len(line1)):
        sheet3.write(0,j,line1[j])
    samples=[]
    file=f.read()
    records=file.split('\n')
    for y in range(len(records)-1):
        samples.append(records[y].split('\t'))

    fo = open(f'v20220626/output/{perfix_name}_{i}/64_1e-06/test.txt', 'r')
    pre = []
    File = fo.read()
    Records = File.split('\n')
    for y in range(0, len(Records) - 1):
        pre.append(Records[y].split('\t'))  # 把表格数据存入
    for j in range(len(participants[i])-1):
        for z in range(1,4):
            cw=0
            zs=0
            sheet3.write(3*j+z,0,participants[i][j]+str(z))
            lie=0
            for k in range(1,14):
                if k !=3 and k!=6 and k!=12:
                    lie+=1
                    xr=''
                    for y in range(len(samples)):
                        if samples[y][0]=='media'+str(k) and samples[y][1]==participants[i][j]+str(z):

                            zs+=1
                            pzs[lie-1]+=1
                            sezs[z - 1]+=1
                            xr=str(pre[y][0])+str(pre[y][1])
                            if str(pre[y][0])!=str(pre[y][1]):
                                cw+=1
                                pcw[lie-1]+=1
                                secw[z-1]+=1
                            break
                    sheet3.write(3*j+z,lie,xr)
            sheet3.write(3*j+z,11,str(cw)+'/'+str(zs))
            if cw==0:
                sheet3.write(3 * j + z, 12, 0)
            else:
                sheet3.write(3*j+z,12,np.round(cw/zs,4))
    for m in range(1,11):
        sheet3.write(3*j+z+1,m,str(pcw[m-1])+'/'+str(pzs[m-1]))
        if pcw[m-1]==0:
            sheet3.write(3*j+z+2, m, 0)
        else:
            sheet3.write(3*j+z+2, m, np.round(pcw[m-1] / pzs[m-1], 4))
    sheet.save(f'v20220626/output/{perfix_name}_{i}/{i}.xls')
    print(secw/sezs)
