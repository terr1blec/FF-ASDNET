import numpy as np
import pandas as pd
import os
import xlwt
from multiprocessing import Process
import openpyxl



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
        self.crudegazepoint='3crudegazepoint'

        self.oksaccadefixation = '5oksaccadefixation'
        self.okfixation = '5okfixation'
        self.okgazepoint = '5okgazepoint'
        self.okcrudegazepoint='5okcrudegazepoint'
        self.videostartsplit = 'videostartsplit'

        #处理完的一些样本是不能用的，能用的样本在这里
        self.fixationsample='6fixationsample'
        self.saccadefixationsample='6saccadefixationsample'
        self.gazepointsample='6gazepointsample'
        self.crudegazepointsample='6crudegazepointsample'

        self.fixationimgcoor='fixationimg/fixationcoor'
        self.saccadefixationimgcoor = 'saccadefixationimg/fixationcoor'
        self.gazepointimgcoor = 'gazepointimg/fixationcoor'
        self.crudegazepointimgcoor='gazepointimg/crudegazepoint'
        self.delecrudegazepointimgcoor = 'gazepointimg/delecrudegazepoint'
        self.missingsamples='gazepointimg/missingsamples'
        self.xy='gazepointimg/indexxy'
        self.gazepointfps='gazepointfps'

pa=paramas()
sequence=pd.read_excel('视频播放顺序.xlsx')
sequence=sequence.values.tolist()
sequence=np.delete(sequence,[36,37],0)

samples={}
features={}
for i in range(1,15):
    samples['media'+str(i)]=[]
    features['media' + str(i)] = []
    files=os.listdir(f'{pa.delecrudegazepointimgcoor}0.5/media{i}')
    for file in files:
        samples['media'+str(i)].append(file)
    sheet=pd.read_excel(f'stafeatures/stafeatureswithmeand/media{i}.xls')
    sheet=sheet.values.tolist()
    # sheet=np.delete(sheet,[17,18],1)
    for j in range(len(sheet)):
        re=[sheet[j][0],sheet[j][30],sheet[j][31],sheet[j][3],sheet[j][4],sheet[j][34],sheet[j][35],sheet[j][36],sheet[j][37],sheet[j][38],sheet[j][39],
            sheet[j][40],sheet[j][41],sheet[j][42],sheet[j][43],sheet[j][15],sheet[j][16],sheet[j][48],sheet[j][25],sheet[j][20],sheet[j][21],sheet[j][22],sheet[j][23],sheet[j][24]]
        # features['media'+str(i)].append(sheet[j][:19])
        features['media' + str(i)].append(re)


data=[]
data.append(['Participant','isASD','media','happy','sequence','happy_strong','fixations',	'duration','m_duration','sd_duration','fixa','fixb','fixc','fixd',
             'fixe','dura','durb','durc','durd','dure','mdisimg','sddisimg','revisits','firstfixationtime','meanda','meandb','meandc','meandd','meande'])
for i in range(len(sequence)):
    if sequence[i][0][0]=='n':
        isasd=2
    else:
        isasd=1
    for j in range(1,15):
        if j>7:
            happy=1
            if j<11:
                happy_strong=11
            else:
                happy_strong=12
        else:
            happy=2
            if j<4:
                happy_strong=21
            else:
                happy_strong=22
        print(len(samples['media'+str(j)]))
        print(len(features['media'+str(j)]))
        for k in range(len(features['media'+str(j)])):
            if features['media'+str(j)][k][0][:-1]==sequence[i][0]:
                # print(1)
                if features['media'+str(j)][k][0]+'.xlsx' in samples['media'+str(j)]:
                    record=[sequence[i][0],isasd,j,happy,int(float(sequence[i][int(features['media'+str(j)][k][0][len(features['media'+str(j)][k][0])-1])])),happy_strong]
                    record.extend(features['media'+str(j)][k][1:])
                    data.append(record)

f = xlwt.Workbook()
sheet3 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
for i in range(len(data)):
    for j in range(len(data[i])):
        sheet3.write(i,j,data[i][j])
f.save(f'spssdata17withmeand.xls')
