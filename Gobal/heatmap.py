import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
import model
from dataloader import pallmediaMyDataset


# 图片预处理
def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]  # 1
    img = np.ascontiguousarray(img)  # 2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # 3
    return img


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
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.2 * heatmap + 0.8 * img

    cv2.imwrite(out_dir, cam_img)

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
        self.version='v20220819/saliencyaddgazemmwos4'
        # self.version = 'v20220819/dvmm3'
        # self.version = 'v20221024/saliencyadddv'
        # self.version = 'v20221108/saliencyaddgazemmwos4'

if __name__ == '__main__':

    args=parameter()
    samplepath = '../sample/20220817wos4'
    # imgpath = '../DataProcessing/gazepointimg/gazepointimgmmwos415'
    # prefix_name = f'14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle'
    # batch = 64
    # lr = 1e-6
    # ws=8
    # best_epoch = 100

    imgpath = '../DataProcessing/saliencyaddgazemmwos4'
    prefix_name = '14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle'
    batch = 16
    lr = 1e-6
    best_epoch = 80
    ws = 7

    resnet = model.resnet(True)
    _ = resnet.eval()
    # for name, module in resnet._modules.items():
    #     for parma in module.parameters():
    #         parma.requires_grad = False

    for round in range(5):
        pathfiles = os.listdir(f'{args.version}/models/{prefix_name}/{round}/{batch}_{lr}')
        for file in pathfiles:
            if file[5:ws] == str(best_epoch):
                pathfile = f'{args.version}/models/{prefix_name}/{round}/{batch}_{lr}/{file}'
        resnet.load_state_dict(torch.load(pathfile))

        validation_dataset = pallmediaMyDataset(args, imgpath, samplepath, round, train='test')
        samples = validation_dataset.getsamples()
        labels=validation_dataset.getlabel()
        for i in range(len(samples)):
            array = cv2.imread(f'{imgpath}/{round}/{samples[i][0]}/{samples[i][1]}.jpg')
            img_tensor = validation_dataset.transform(array)
            img_tensor = img_tensor.unsqueeze(0)

            # 存放梯度和特征图
            fmap_block = list()
            grad_block = list()

            # 注册hook
            # set_trace()
            # net.features[-1].expand3x3.register_forward_hook(farward_hook)	# 9
            # net.features[-1].expand3x3.register_backward_hook(backward_hook)

            resnet.model.layer4[-1].register_forward_hook(farward_hook)  # 9
            resnet.model.layer4[-1].register_backward_hook(backward_hook)

            # forward
            output = resnet(img_tensor)
            idx = np.argmax(output.cpu().data.numpy())
            # print("predict: {}".format(classes[idx]))

            # backward
            resnet.zero_grad()
            loss_fn = torch.nn.CrossEntropyLoss()
            # loss_fn=loss_fn.cuda()
            loss = loss_fn(output, torch.tensor([labels[i]]))
            loss.backward()
            probas = torch.softmax(output, dim=1)
            pre = torch.argmax(probas, dim=1)

            # 生成cam
            grads_val = grad_block[0].cpu().data.numpy().squeeze()
            fmap = fmap_block[0].cpu().data.numpy().squeeze()

            # 保存cam图片
            if not os.path.exists(f'{prefix_name}_2/cam/{round}/{samples[i][0]}'):
                os.makedirs(f'{prefix_name}_2/cam/{round}/{samples[i][0]}')
            output_dir = f'{prefix_name}_2/cam/{round}/{samples[i][0]}/{samples[i][1]}_{pre[0]}.jpg'
            cam_show_img(array, fmap, grads_val, output_dir)
