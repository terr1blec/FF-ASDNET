import os

import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch
import torch.nn.functional as F
import cv2

import model
from dataloader import pallmediaMyDataset

# ----------------------------------- feature map visualization -----------------------------------

writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

# 注册hook
fmap_dict = dict()
n = 0

class parameter():
    def __init__(self):
        self.lr_scheduler='plateau'
        self.gamma=0.5
        self.epochs=100
        self.lr=1e-6
        self.batch_size=32
        self.flush_history=0
        self.patience=5
        self.save_model=1
        self.log_every=100
        # self.version='v20220819/addgazemmwos4'
        # self.version = 'v20220819/dvmm3'
        self.version = 'v20220819/saliencyaddgazemmwos4'
        # self.version = 'v20221024/saliencyadddv'

def hook_func(m, i, o):
    key_name = str(m.weight.shape)
    fmap_dict[key_name].append(o)

args = parameter()
samplepath = '../sample/20220817wos4'

# imgpath = '../DataProcessing/gazepointimg/gazepointimgmmwos415'
# prefix_name = f'14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle'
# batch = 64
# lr = 1e-6
# ws=8
# best_epoch = 100

# imgpath= '../DataProcessing/DynamicalVisualize/img0913'
# prefix_name='14resnet_3fc_rgb17wos4/d0.5_n0.5_shuffle'
# ws=7
# best_epoch=80
# batch = 64
# lr = 1e-6

imgpath='../DataProcessing/saliencyaddgazemmwos4'
prefix_name='14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle'
batch=16
lr=1e-6
best_epoch=80
ws=7
resnet = model.resnet(True)
_ = resnet.eval()
for name,module in resnet._modules.items():
    for parma in module.parameters():
        parma.requires_grad=False
for name, sub_module in resnet.named_modules():
    if isinstance(sub_module, nn.Conv2d):
        n += 1
        key_name = str(sub_module.weight.shape)
        fmap_dict.setdefault(key_name, list())
        n1 = name.split(".")
        if len(n1)==1:
            resnet._modules[n1].register_forward_hook(hook_func)
        elif len(n1)==2:
            resnet._modules[n1[0]]._modules[n1[1]].register_forward_hook(hook_func)
        elif len(n1)==3:
            resnet._modules[n1[0]]._modules[n1[1]]._modules[n1[2]].register_forward_hook(hook_func)
        elif len(n1)==4:
            resnet._modules[n1[0]]._modules[n1[1]]._modules[n1[2]]._modules[n1[3]].register_forward_hook(hook_func)
# for round in range(5):
#     pathfiles = os.listdir(f'{args.version}/models/{prefix_name}/{round}/{batch}_{lr}')
#     for file in pathfiles:
#         if file[5:ws] == str(best_epoch):
#             pathfile = f'{args.version}/models/{prefix_name}/{round}/{batch}_{lr}/{file}'
#     resnet.load_state_dict(torch.load(pathfile))
#
#
#     validation_dataset = pallmediaMyDataset(args, imgpath, samplepath, round, train='test')
#     samples = validation_dataset.getsamples()
#     # for sample in samples:
#     array = cv2.imread(f'{imgpath}/{round}/{samples[0][0]}/{samples[0][1]}.jpg')
#     img_tensor = validation_dataset.transform(array)
#     img_tensor = img_tensor.unsqueeze(0)
#     # forward
#     output = resnet(img_tensor)
#     # print(fmap_dict['torch.Size([128, 64, 3, 3])'][0].shape)
#     # add image
#     for layer_name, fmap_list in fmap_dict.items():
#         try :
#             fmap = fmap_list[0]
#             # print(fmap.shape)
#             fmap.transpose_(0, 1)
#             # print(fmap.shape)
#
#             nrow = int(np.sqrt(fmap.shape[0]))
#             # if layer_name == 'torch.Size([512, 512, 3, 3])':
#
#             fmap = F.interpolate(fmap, size=[112, 112], mode="bilinear")
#             fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
#             # print(type(fmap_grid), fmap_grid.shape)
#             writer.add_image('round{} {} {} feature map in {}'.format(round,samples[0][0],samples[0][1],layer_name), fmap_grid, global_step=322)
#         except:
#             print('')

pathfiles = os.listdir(f'{args.version}/models/{prefix_name}/0/{batch}_{lr}')
for file in pathfiles:
    if file[5:ws] == str(best_epoch):
        pathfile = f'{args.version}/models/{prefix_name}/0/{batch}_{lr}/{file}'
resnet.load_state_dict(torch.load(pathfile))
array = cv2.imread(f'{imgpath}/0/media1/a006lzq1.jpg')
validation_dataset = pallmediaMyDataset(args, imgpath, samplepath, 0, train='test')
img_tensor = validation_dataset.transform(array)
img_tensor = img_tensor.unsqueeze(0)
 # forward
output = resnet(img_tensor)
# print(fmap_dict['torch.Size([128, 64, 3, 3])'][0].shape)
# add image
for layer_name, fmap_list in fmap_dict.items():
    try :
        fmap = fmap_list[0]
        # print(fmap.shape)
        fmap.transpose_(0, 1)
        # print(fmap.shape)

        nrow = int(np.sqrt(fmap.shape[0]))
        # if layer_name == 'torch.Size([512, 512, 3, 3])':

        fmap = F.interpolate(fmap, size=[112, 112], mode="bilinear")
        fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
        # print(type(fmap_grid), fmap_grid.shape)
        writer.add_image('round media1 a006lzq1 feature map in {}'.format(layer_name), fmap_grid, global_step=322)
    except:
        print('')
