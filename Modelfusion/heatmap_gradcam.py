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

# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)


# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 7

    # 将CAM值缩放到0到1的范围
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (W, H))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # cam = np.maximum(cam, 0)
    # print(cam.max())
    # cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    # 2024-05-08 王子霖
    # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # cam = 0.3 * heatmap + 0.7 * np.float32(img)
    # cam = cam / np.max(cam)
    # cam_img = np.uint8(255 * cam)

    cv2.imwrite(out_dir, cam_img)

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
    lr = 1e-6
    best_epoch = 20
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
    # zsallpath = f'{zsmodelpath}/{i}/64_1e-06'
    # zsfiles = os.listdir(zsallpath)
    # for zsfile in zsfiles:
    #     if zsfile[5:8] == str(100):
    #         zspath = f'{zsmodelpath}/{i}/64_1e-06/{zsfile}'
    #         break
    zsallpath = f"{zsmodelpath}/{i}/64_1e-06"
    zsfiles = os.listdir(zsallpath)
    for zsfile in zsfiles:
        if zsfile[5:7] == str(80):  # 之前是zsfile[5:7] == str(80):
            zspath = f"{zsmodelpath}/{i}/64_1e-06/{zsfile}"
            break

    # lallpath = f'{lmodelpath}/{i}/16_6e-07'
    lallpath = f'{lmodelpath}/{i}/32_1e-06'
    lfiles = os.listdir(lallpath)
    for lfile in lfiles:
        if lfile[5:7] == str(40):
            lpath = f'{lmodelpath}/{i}/32_1e-06/{lfile}'
            break

    dvzsallpath = f"{dvzs433modelpath}/{i}/64_1e-06"
    dvzsfiles = os.listdir(dvzsallpath)
    for dvzsfile in dvzsfiles:
        # if dvzsfile[5:7] == str(20):
        if dvzsfile[5:7] == str(20):  # 之前是40
            dvzspath = f"{dvzs433modelpath}/{i}/64_1e-06/{dvzsfile}"
            break

    resnet = model.dvzsfusionconv_lf_fccon(dvpath,zspath,lpath=lpath,dvzspath=dvzspath,update=True,fc=5,d=0.5,padding=1)
    resnet.cuda()
    resnet.eval()
    # for name, module in resnet._modules.items():
    #     for parma in module.parameters():
    #         parma.requires_grad = False

    for round in range(5):
        pathfiles = os.listdir(f'{args.version}/models/{prefix_name}/{round}/{batch}_{lr}')
        for file in pathfiles:
            if file[5:ws] == str(best_epoch):
                pathfile = f'{args.version}/models/{prefix_name}/{round}/{batch}_{lr}/{file}'
        resnet.load_state_dict(torch.load(pathfile))


        validation_dataset = pallmediaMyDataset(args, dvimgpath,zsimgpath,limgpath,True, samplepath, round, train='trainvalidation',padding=1) #test
        # validation_loader = torch.utils.data.DataLoader(
        #     validation_dataset, batch_size=32, shuffle=False, num_workers=16, drop_last=False, pin_memory=False,
        #     collate_fn=_collate_fn)
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


            # print(resnet)
            resnet.dvzsmodel.zsmodel.model.layer4[-1].register_forward_hook(farward_hook)  # 9
            resnet.dvzsmodel.zsmodel.model.layer4[-1].register_full_backward_hook(backward_hook)
            # resnet.dvzsmodel.dvmodel.model.layer4[-1].register_forward_hook(farward_hook)  # 9
            # resnet.dvzsmodel.dvmodel.model.layer4[-1].register_full_backward_hook(backward_hook)
            # resnet.lmodel.layer4[-1].register_forward_hook(farward_hook)  # 9
            # resnet.lmodel.layer4[-1].register_full_backward_hook(backward_hook)
            # forward
            bz1=[1.0]
            bz=np.array(bz1)
            output = resnet(dvimg_tensor,zsimg_tensor,limg_tensor,bz,0)
            idx = np.argmax(output.cpu().data.numpy())
            # print("predict: {}".format(classes[idx]))

            # backward
            resnet.zero_grad()
            loss_fn = torch.nn.CrossEntropyLoss()
            # loss_fn=loss_fn.cuda()
            real=torch.tensor([labels[i]])
            real=real.cuda()
            loss = loss_fn(output, real)
            loss.backward()
            probas = torch.softmax(output, dim=1)
            pre = torch.argmax(probas, dim=1)

            # 生成cam
            grads_val = grad_block[0].cpu().data.numpy().squeeze()
            fmap = fmap_block[0].cpu().data.numpy().squeeze()

            # 保存cam图片
            if not os.path.exists(f'zsheatmap/train/{args.version}/{round}/{samples[i][0]}'):
                os.makedirs(f'zsheatmap/train/{args.version}/{round}/{samples[i][0]}')
            output_dir = f'zsheatmap/train/{args.version}/{round}/{samples[i][0]}/{samples[i][1]}_{pre[0]}.jpg'
            cam_show_img(zsarray, fmap, grads_val, output_dir)

            # if not os.path.exists(f'map/lmodelheatmap/train/{args.version}_{lr}_{best_epoch}/{round}/{samples[i][0]}'):
            #     os.makedirs(f'map/lmodelheatmap/train/{args.version}_{lr}_{best_epoch}/{round}/{samples[i][0]}')
            # output_dir = f'map/lmodelheatmap/train/{args.version}_{lr}_{best_epoch}/{round}/{samples[i][0]}/{samples[i][1]}_{pre[0]}.jpg'
            # cam_show_img(larray, fmap, grads_val, output_dir)

            # if not os.path.exists(f'map/dvheatmap/train/{args.version}/{round}/{samples[i][0]}'):
            #     os.makedirs(f'map/dvheatmap/train/{args.version}/{round}/{samples[i][0]}')
            # output_dir = f'map/dvheatmap/train/{args.version}/{round}/{samples[i][0]}/{samples[i][1]}_{pre[0]}.jpg'
            # cam_show_img(dvarray, fmap, grads_val, output_dir)