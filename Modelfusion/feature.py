import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
import model
from dataloader import pallmediaMyDataset,_collate_fn
import torch.nn as nn
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image

class parameter():
    def __init__(self):
        self.lr_scheduler='plateau'
        self.gamma=0.5
        self.epochs=50
        self.lr=1e-6
        self.batch_size=32
        self.flush_history=0
        self.patience=5
        self.save_model=1
        self.log_every=100
        self.version='v20230320/dv_zssa_l'

if __name__ == '__main__':

    args=parameter()
    samplepath = '../sample/20220817wos4'

    dvimgpath = '../DataProcessing/DynamicalVisualize/imgwos4_4'
    zsimgpath = '../DataProcessing/saliencyaddgazemmwos4'
    limgpath = '../DataProcessing/gazepointimg/fsjoint1d12350'
    prefix_name = f'l_5fc_padding_d0.5_convcon_conv512_1*1_conv256_3*3_update_1loss_3fc/d0.5_n0.5_shuffle'
    batch = 64
    lr = 5e-7
    best_epoch = 50
    ws = 7

    dvmodelpath='../globalheatmapclassifier/v20230223/dvmm/models/14resnet_3fc_rgb17wos4_4/d0.5_n0.5_shuffle'
    zsmodelpath='../globalheatmapclassifier/v20220819/saliencyaddgazemmwos4/models/14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle'
    lmodelpath='../exp20220616/longimg/v20230316/models/conv2dconv1d_rconv1d1024(321)_2048(521)_17wos4_BN_3fc_rgb_d0.5_shuffle_'
    dvzs433modelpath = 'v20230226/dv_zssa/models/convcon_conv512_1*1_conv256_3*3_update_1loss_3fc/d0.5_n0.5_shuffle'

    i=0
    print(i)
    dvallpath=f'{dvmodelpath}/{i}/64_1e-06'
    dvfiles=os.listdir(dvallpath)
    for dvfile in dvfiles:
        if dvfile[5:8]==str(100):
            dvpath=f'{dvmodelpath}/{i}/64_1e-06/{dvfile}'
            break

    zsallpath = f"{zsmodelpath}/{i}/64_1e-06"
    zsfiles = os.listdir(zsallpath)
    for zsfile in zsfiles:
        if zsfile[5:7] == str(80):  # 之前是zsfile[5:7] == str(80):
            zspath = f"{zsmodelpath}/{i}/64_1e-06/{zsfile}"
            break

    lallpath = f'{lmodelpath}/{i}/32_1e-06'
    lfiles = os.listdir(lallpath)
    for lfile in lfiles:
        if lfile[5:7] == str(40):
            lpath = f'{lmodelpath}/{i}/32_1e-06/{lfile}'
            break

    dvzsallpath = f"{dvzs433modelpath}/{i}/64_1e-06"
    dvzsfiles = os.listdir(dvzsallpath)
    for dvzsfile in dvzsfiles:
        if dvzsfile[5:7] == str(20):  # 之前是40
            dvzspath = f"{dvzs433modelpath}/{i}/64_1e-06/{dvzsfile}"
            break

    resnet = model.dvzsfusionconv_lf_fccon(dvpath,zspath,lpath=lpath,dvzspath=dvzspath,update=True,fc=5,d=0.5,padding=1)
    resnet.cuda()
    _ = resnet.eval()

    for round in range(5):
        pathfiles = os.listdir(f'{args.version}/models/{prefix_name}/{round}/{batch}_{lr}')
        for file in pathfiles:
            if file[5:ws] == str(best_epoch):
                pathfile = f'{args.version}/models/{prefix_name}/{round}/{batch}_{lr}/{file}'
        resnet.load_state_dict(torch.load(pathfile))

        cam_extractor = SmoothGradCAMpp(resnet,target_layer = resnet.dvzsmodel.dvmodel.model.layer4[-1])

        validation_dataset = pallmediaMyDataset(args, dvimgpath,zsimgpath,limgpath,True, samplepath, round, train='trainvalidation',padding=1) #test

        samples = validation_dataset.getsamples()
        labels=validation_dataset.getlabel()
        for i in range(len(samples)):
            zsarray = cv2.imread(f'{zsimgpath}/{round}/{samples[i][0]}/{samples[i][1]}.jpg')
            zsimg_tensor = validation_dataset.transform(zsarray)
            zsimg_tensor = zsimg_tensor.unsqueeze(0)

            dvarray = cv2.imread(f'{dvimgpath}/{round}/{samples[i][0]}/{samples[i][1]}.jpg')
            dvimg_tensor = validation_dataset.transform(dvarray)
            dvimg_tensor = dvimg_tensor.unsqueeze(0)

            larray = cv2.imread(f'{limgpath}/{samples[i][0]}/{samples[i][1]}.jpg')
            limg_tensor = validation_dataset.ltransform(larray)
            limg_tensor = limg_tensor.unsqueeze(0)

            zsimg_tensor=zsimg_tensor.cuda()
            dvimg_tensor=dvimg_tensor.cuda()
            limg_tensor=limg_tensor.cuda()

            # 存放梯度和特征图
            fmap_block = list()
            grad_block = list()

            # forward
            bz1=[1.0]
            bz=np.array(bz1)
            out = resnet(dvimg_tensor,zsimg_tensor,limg_tensor,bz,0)
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
            result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            
            # 保存cam图片
            if not os.path.exists(f'try/{round}/{samples[i][0]}'):
                os.makedirs(f'try/{round}/{samples[i][0]}')
            output_dir = f'try/{round}/{samples[i][0]}/{samples[i][1]}_{pre[0]}.jpg'
            cv2.imwrite(output_dir , result )
            
            # if not os.path.exists(f'lmodelheatmap/train/{args.version}/{round}/{samples[i][0]}'):
            #     os.makedirs(f'lmodelheatmap/train/{args.version}/{round}/{samples[i][0]}')
            # output_dir = f'lmodelheatmap/train/{args.version}/{round}/{samples[i][0]}/{samples[i][1]}_{pre[0]}.jpg'
            # cam_show_img(larray, fmap, grads_val, output_dir)
            # if not os.path.exists(f'dvheatmap/train/{args.version}/{round}/{samples[i][0]}'):
            #     os.makedirs(f'dvheatmap/train/{args.version}/{round}/{samples[i][0]}')
            # output_dir = f'dvheatmap/train/{args.version}/{round}/{samples[i][0]}/{samples[i][1]}_{pre[0]}.jpg'
            # cam_show_img(dvarray, fmap, grads_val, output_dir)