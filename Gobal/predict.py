import torch
import torchvision.transforms as transforms
import cv2
from model import resnet
import shutil
import os
import numpy as np



def pre(imgpath,resultpath,file,net):
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize([224, 224]),
                                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 导入要测试的图像（自己找的一张猫的图片，不在数据集中），放在testImage目录下
    im = cv2.imread(f'{imgpath}/{file[0]}/{file[1]}.jpg')
    # im = cv2.imread(f'{imgpath}/testpic.jpg')
    im = transform(im)  # [C, H, W]
    im = im.unsqueeze(0)  # 对数据增加一个新维度，因为tensor的参数是[batch, channel, height, width]
    # 预测
    classes = ('TD','ASD')
    with torch.no_grad():
        outputs = net.forward(im)
        probas = torch.softmax(outputs, dim=1)
        predict = torch.argmax(probas, dim=1).data.numpy()
    print(predict,probas)
    print("*********************************"+str(file))
    if file[1][0]=='n':
        ASD='TD'
    else:
        ASD='ASD'

    if classes[int(predict[0])]==ASD:
        right='right'
    else:
        right='wrong'

    # shutil.copy(f'{imgpath}/{file[0]}/{file[1]}.jpg',f'{resultpath}/{right}/{file[0]}/{ASD}/{file[1]}.jpg')
    return ASD,right


# 实例化网络，加载训练好的模型参数
#modelpath='v20220819/saliencyaddgazemmwos4/models/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle/2/16_1e-06/test_epoch80_trainacc0.8635_valacc0.6461.pth'
modelpath='v20230223/dvmm/models/14resnet_3fc_rgb17wos4_4/d0.5_n0.5_shuffle/0/64_1e-06/epoch100_trainacc0.7755_valacc0.6135.pth'
net = resnet(True)
net.load_state_dict(torch.load(modelpath))
net.eval()
imgpath='../DataProcessing/saliencyaddgazemmwos4'
resultpath='predict/0'
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

for i in range(1,15):
    if not os.path.exists(f'{resultpath}/right/media{i}/ASD'):
        os.makedirs(f'{resultpath}/right/media{i}/ASD')
    if not os.path.exists(f'{resultpath}/wrong/media{i}/ASD'):
        os.makedirs(f'{resultpath}/wrong/media{i}/ASD')
    if not os.path.exists(f'{resultpath}/right/media{i}/TD'):
        os.makedirs(f'{resultpath}/right/media{i}/TD')
    if not os.path.exists(f'{resultpath}/wrong/media{i}/TD'):
        os.makedirs(f'{resultpath}/wrong/media{i}/TD')

fo = open(f'../sample/20220817wos4/14/2/testsample.txt', 'r')
participants = []
myFile = fo.read()
myRecords = myFile.split('\n')
print("^^^^^^^^^^^^^^^^^^^^^"+str(myRecords))
for y in range(0, len(myRecords) - 1):
    participants.append(myRecords[y].split('\t'))  # 把表格数据存入
# for i in range(len(participants)):
#     pre(imgpath,resultpath,participants[i],net)
#
# # fo = open(f'v20220819/saliencyaddgazemm/output/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle_0/32_1e-06/test_epoch100_trainacc0.8901_valacc0.6155.txt', 'r')
# # fo = open(f'v20220819/saliencyaddgazemm/output/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle_1/32_1e-06/test_epoch100_trainacc0.9006_valacc0.6019.txt', 'r')
# # fo = open(f'v20220819/saliencyaddgazemm/output/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle_2/32_1e-06/test_epoch100_trainacc0.8842_valacc0.6393.txt', 'r')
# # fo = open(f'v20220819/saliencyaddgazemm/output/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle_3/32_1e-06/test_epoch100_trainacc0.8942_valacc0.6321.txt', 'r')
fo = open(f'v20220819/saliencyaddgazemmwos4/output/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle_2/16_1e-06/test_epoch80_trainacc0.8635_valacc0.6461.txt', 'r')
val = []
myFile = fo.read()
myRecords = myFile.split('\n')
for y in range(0, len(myRecords) - 1):
    val.append(myRecords[y].split('\t'))  # 把表格数据存入

samples=len(participants)
sampleright=0
samplewrong=0
allASD=0
allTD=0
ASDright=0
TDright=0
ASDwrong=0
TDwrong=0
mediasample= {}
mediaASD={}
mediaTD={}
mediawrong={}
mediaright={}
mediaASDwrong={}
mediaASDright={}
mediaTDwrong={}
mediaTDright={}
roundsamples={}
roundTD={}
roundASD={}
roundTDright={}
roundTDwrong={}
roundASDright={}
roundASDwrong={}
for i in range(1,4):
    roundsamples[str(i)] = 0
    roundTD[str(i)]=0
    roundASD[str(i)] = 0
    roundASDwrong[str(i)] = 0
    roundASDright[str(i)] = 0
    roundTDwrong[str(i)] = 0
    roundTDright[str(i)] = 0
for i in range(1,15):
    mediasample['media'+str(i)]=0
    mediaASDright['media'+str(i)]=0
    mediawrong['media'+str(i)]=0
    mediaTDwrong['media'+str(i)]=0
    mediaright['media'+str(i)]=0
    mediaASDwrong['media'+str(i)]=0
    mediaTD['media'+str(i)]=0
    mediaASD['media'+str(i)]=0
    mediaTDright['media'+str(i)]=0


for i in range(len(participants)):
    mediasample[participants[i][0]]+=1
    roundsamples[participants[i][1][-1]]+=1
    ASD,right=pre(imgpath,resultpath,participants[i],net)
    if right=='right':
        sampleright+=1
    else:
        samplewrong+=1
    if ASD=='ASD':
        allASD+=1
        mediaASD[participants[i][0]]+=1
        roundASD[participants[i][1][-1]]+=1
        if right=='right':
            ASDright+=1
            mediaASDright[participants[i][0]]+=1
            roundASDright[participants[i][1][-1]]+=1
        else:
            ASDwrong+=1
            mediaASDwrong[participants[i][0]]+=1
            roundASDwrong[participants[i][1][-1]]+=1
    else:
        allTD+=1
        mediaTD[participants[i][0]]+=1
        roundTD[participants[i][1][-1]]+=1
        if right=='right':
            TDright+=1
            mediaTDright[participants[i][0]]+=1
            roundTDright[participants[i][1][-1]]+=1
        else:
            TDwrong+=1
            mediaTDwrong[participants[i][0]]+=1
            roundTDwrong[participants[i][1][-1]]+=1


##########################################################################


# for i in range(len(participants)):
#     mediasample[participants[i][0]]+=1
#     roundsamples[participants[i][1][-1]]+=1

#     if val[i][0]==str(0):
#         ASD='TD'
#     else:
#         ASD='ASD'
#     if val[i][0]==val[i][1]:
#         right='right'
#     else:
#         right='wrong'

#     if right=='right':
#         sampleright+=1
#         mediaright[participants[i][0]]+=1
#     else:
#         samplewrong+=1
#     if ASD=='ASD':
#         allASD+=1
#         mediaASD[participants[i][0]]+=1
#         roundASD[participants[i][1][-1]]+=1
#         if right=='right':
#             ASDright+=1
#             mediaASDright[participants[i][0]]+=1
#             roundASDright[participants[i][1][-1]]+=1
#         else:
#             ASDwrong+=1
#             mediaASDwrong[participants[i][0]]+=1
#             roundASDwrong[participants[i][1][-1]]+=1
#     else:
#         allTD+=1
#         mediaTD[participants[i][0]]+=1
#         roundTD[participants[i][1][-1]]+=1
#         if right=='right':
#             TDright+=1
#             mediaTDright[participants[i][0]]+=1
#             roundTDright[participants[i][1][-1]]+=1
#         else:
#             TDwrong+=1
#             mediaTDwrong[participants[i][0]]+=1
#             roundTDwrong[participants[i][1][-1]]+=1

# print(f'samples: {samples}| ASD: {allASD}| TD: {allTD}')
# print(f'round 1: {roundsamples[str(1)]}| round 2: {roundsamples[str(2)]}| round 3: {roundsamples[str(3)]}')
# print(f'right sample: {sampleright} (right/all {np.round(sampleright/samples,4)})')
# # print(f'wrong sample: {samplewrong} (wrong/all {np.round(samplewrong/samples,4)})')
# print(f'right ASD sample: {ASDright} (ASD:right/all {np.round(ASDright/allASD,4)})')
# print(f'right TD sample: {TDright} (TD:right/all {np.round(TDright/allTD,4)})')
# # print(f'wrong ASD sample: {ASDwrong} (ASD:wrong/all {np.round(ASDwrong/allASD,4)})')
# # print(f'wrong TD sample: {TDwrong} (TD:wrong/all {np.round(TDwrong/allTD,4)})')
# for i in range(1,4):
#     print(f'round{i}: ASD {roundASD[str(i)]}({np.round(roundASD[str(i)]/roundsamples[str(i)],4)})| ASDringht {roundASDright[str(i)]}({np.round(roundASDright[str(i)]/roundASD[str(i)],4)})|TD {roundTD[str(i)]}({np.round(roundTD[str(i)]/roundsamples[str(i)],4)})| TDringht {roundTDright[str(i)]}({np.round(roundTDright[str(i)]/roundTD[str(i)],4)})')
for i in range(1,15):
    media='media'+str(i)
    print(f'{media}: samples {mediasample[media]}|Acc:{np.round(mediaright[media]/mediasample[media],4)} ASD {mediaASD[media]}| ASDringht {mediaASDright[media]}({np.round(mediaASDright[media]/mediaASD[media],4)})|TD {mediaTD[media]}| TDringht {mediaTDright[media]}({np.round(mediaTDright[media]/mediaTD[media],4)})')