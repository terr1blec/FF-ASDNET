import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import random
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor

class resnet(nn.Module):
    def __init__(self,pre):
        super().__init__()
        self.model=models.resnet18(pretrained=pre) #使用预训练好的模型参数
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512,256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128,2))

    def forward(self,x):
        n=self.model
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        x = n.layer4(x)
        pooled_features = self.pooling_layer(x)

        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifier(pooled_features)
        return output

class lresnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=models.resnet18(pretrained=True) #使用预训练好的模型参数
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 2))

    def forward(self,x,bz,device):
        n=self.model
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        x = n.layer4(x)

        pooled_features = self.pooling_layer(x)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifer(pooled_features)
        return output

def conv3x3(in_planes: int, out_planes: int, kernel_size,padding,stride:int=1, groups: int = 1, dilation: int=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int,kernel_size, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel: tuple,
        stride: int=1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int=1,
        padding: tuple=(0,0),
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes,kernel,padding, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,kernel,padding)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class conv2dconv1drpadding(nn.Module):
    def __init__(
        self,
        layers: List[int],
        num_classes: int = 2,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=(3,11), stride=2, padding=(1,5), bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,9), stride=(1,2), padding=(0,4))
        self.layer1 = self._make_layer( 64, layers[0],stride=1,kernel=(1,9),padding=(0,4))
        self.layer2 = self._make_layer( 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],kernel=(1,9),padding=(0,4))
        self.layer3 = self._make_layer(256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],kernel=(1,9),padding=(0,4))
        self.layer4 = self._make_layer( 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],kernel=(1,9),padding=(0,4))
        self.pooling_layer1 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.Dropout(0.5),
            # nn.Linear(64, 32),
            # nn.Dropout(0.2),
            nn.Linear(128,num_classes))
        self.conv1d = nn.Sequential(
            nn.Conv1d(512, 1024, 3, 2, 1, bias=False),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 2048, 5, 2, 1, bias=False),
            # nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(2048, 512, 3, 1, 1, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        # block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        kernel: tuple,
        stride: int,
        padding:tuple=(0,0),
        dilate: bool = False,
        dilation:int=1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = dilation
        # if dilate:
        #     self.dilation *= stride
        #     stride = 1
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, 1,stride),
                norm_layer(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(
            BasicBlock(
                self.inplanes, planes, kernel,stride, downsample, self.groups, self.base_width, previous_dilation,padding, norm_layer
            )
        )
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    kernel=kernel,
                    stride=1,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    padding=padding,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, bz, device) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature1 = torch.zeros((x.shape[0], x.shape[1], 1, x.shape[3])).cuda()
        for i in range(x.shape[0]):
            pooling = nn.AdaptiveAvgPool2d((1, int(x.shape[3] * bz[i])))
            feature1[i, :, :, :int(x.shape[3] * bz[i])] = pooling(x[i, :, :, :int(x.shape[3] * bz[i])])
        pooled_features = torch.squeeze(feature1)
        x1 = self.conv1d(pooled_features)
        output = torch.zeros((x1.shape[0], 2)).cuda()
        for i in range(x1.shape[0]):
            feature2 = self.pooling_layer1(x1[i, :, :int(x1.shape[2] * bz[i])])
            output[i] = self.fc(feature2.view(feature2.shape[0]))
        return output

    def forward(self, x: Tensor, bz, device) -> Tensor:
        return self._forward_impl(x, bz, device)

class dvzsfusionfccon(nn.Module):
    def __init__(self,dvpath,zspath,lpath,update):
        super().__init__()
        self.dvmodel=resnet(True)
        self.zsmodel=resnet(True)
        self.dvmodel.load_state_dict(torch.load(dvpath))
        self.zsmodel.load_state_dict(torch.load(zspath))
        if update==False:
            for name,module in self.dvmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.zsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            # nn.Dropout(0.3),
            nn.Linear(1024, 512),
            # nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            # # # nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(256, 128),
            # # # # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(512, 256),
            # nn.Dropout(0.5),
            nn.Linear(256, 2))

    def forward(self,dvimg,zsimg,limg):
        dv=self.dvmodel
        zs=self.zsmodel

        dvimg = dv.model.conv1(dvimg)
        dvimg = dv.model.bn1(dvimg)
        dvimg = dv.model.relu(dvimg)
        dvimg = dv.model.maxpool(dvimg)
        dvimg = dv.model.layer1(dvimg)
        dvimg = dv.model.layer2(dvimg)
        dvimg = dv.model.layer3(dvimg)
        dvimg = dv.model.layer4(dvimg)
        dvpooled_features = self.pooling_layer(dvimg)
        dvpooled_features = dvpooled_features.view(dvpooled_features.size(0), -1)
        dvoutput=dv.classifier(dvpooled_features)

        zsimg = zs.model.conv1(zsimg)
        zsimg = zs.model.bn1(zsimg)
        zsimg = zs.model.relu(zsimg)
        zsimg = zs.model.maxpool(zsimg)
        zsimg = zs.model.layer1(zsimg)
        zsimg = zs.model.layer2(zsimg)
        zsimg = zs.model.layer3(zsimg)
        zsimg = zs.model.layer4(zsimg)
        zspooled_features = self.pooling_layer(zsimg)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)
        zsoutput=zs.classifier(zspooled_features)

        pooled_features=torch.cat([dvpooled_features, zspooled_features], dim=1)
        output = self.classifier(pooled_features)
        return output,dvoutput,zsoutput

class dvzsfusion_lf_fccon(nn.Module):
    def __init__(self,dvpath,zspath,lpath,dvzspath,update):
        super().__init__()
        self.dvzsmodel=dvzsfusionfccon(dvpath,zspath,lpath,True)
        self.dvzsmodel.load_state_dict(torch.load(dvzspath))
        self.lmodel = lresnet()
        self.lmodel.load_state_dict(torch.load(lpath))
        if update==False:
            for name,module in self.dvzsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.lmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            # nn.Dropout(0.3),
            # nn.Linear(1536, 512),
            # # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(512, 256),
            # # # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(256, 128),
            # # # # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(512, 256),
            # nn.Dropout(0.5),
            nn.Linear(1536, 2))

    def forward(self,dvimg,zsimg,limg):
        dvzs=self.dvzsmodel
        l=self.lmodel

        dvimg = dvzs.dvmodel.model.conv1(dvimg)
        dvimg = dvzs.dvmodel.model.bn1(dvimg)
        dvimg = dvzs.dvmodel.model.relu(dvimg)
        dvimg = dvzs.dvmodel.model.maxpool(dvimg)
        dvimg = dvzs.dvmodel.model.layer1(dvimg)
        dvimg = dvzs.dvmodel.model.layer2(dvimg)
        dvimg = dvzs.dvmodel.model.layer3(dvimg)
        dvimg = dvzs.dvmodel.model.layer4(dvimg)
        dvpooled_features = self.pooling_layer(dvimg)
        dvpooled_features = dvpooled_features.view(dvpooled_features.size(0), -1)
        dvoutput=dvzs.dvmodel.classifier(dvpooled_features)

        zsimg = dvzs.zsmodel.model.conv1(zsimg)
        zsimg = dvzs.zsmodel.model.bn1(zsimg)
        zsimg = dvzs.zsmodel.model.relu(zsimg)
        zsimg = dvzs.zsmodel.model.maxpool(zsimg)
        zsimg = dvzs.zsmodel.model.layer1(zsimg)
        zsimg = dvzs.zsmodel.model.layer2(zsimg)
        zsimg = dvzs.zsmodel.model.layer3(zsimg)
        zsimg = dvzs.zsmodel.model.layer4(zsimg)
        zspooled_features = self.pooling_layer(zsimg)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)
        zsoutput=dvzs.zsmodel.classifier(zspooled_features)

        dvzspooled_features = torch.cat([dvpooled_features, zspooled_features], dim=1)
        dvzsoutput = dvzs.classifier(dvzspooled_features)

        limg = l.model.conv1(limg)
        limg = l.model.bn1(limg)
        limg = l.model.relu(limg)
        limg = l.model.maxpool(limg)
        limg = l.model.layer1(limg)
        limg = l.model.layer2(limg)
        limg = l.model.layer3(limg)
        limg = l.model.layer4(limg)
        lpooled_features = self.pooling_layer(limg)
        lpooled_features = lpooled_features.view(lpooled_features.size(0), -1)
        loutput = l.classifer(lpooled_features)

        pooled_features=torch.cat([dvzspooled_features, lpooled_features], dim=1)
        output = self.classifier(pooled_features)
        return output,loutput,dvzsoutput

class dvzsfusionconv_lf_fccon(nn.Module):
    def __init__(self,dvpath,zspath,lpath,dvzspath,update,fc,d,padding):
        super().__init__()
        self.dvzsmodel=dvzsfusionconvcon_con(dvpath,zspath,lpath,True,0.5)
        if update==True:
            self.dvzsmodel.load_state_dict(torch.load(dvzspath))
        if padding==0:
            self.lmodel = lresnet()
        else:
            self.lmodel=conv2dconv1drpadding([2,2,2,2])
        self.lmodel.load_state_dict(torch.load(lpath))
        self.padding=padding
        if update==False:
            for name,module in self.dvzsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.lmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        if fc==5:
            self.classifier = nn.Sequential(
                nn.Linear(2304, 512),
                nn.Dropout(d),
                nn.Linear(512, 256),
                nn.Dropout(d),
                nn.Linear(256, 128),
                nn.Dropout(d),
                nn.Linear(128, 64),
                nn.Dropout(d),
                nn.Linear(64, 2))
        if fc==4:
            self.classifier = nn.Sequential(
                nn.Linear(2304, 512),
                nn.Dropout(d),
                nn.Linear(512, 256),
                nn.Dropout(d),
                nn.Linear(256, 128),
                nn.Dropout(d),
                nn.Linear(128, 2))
        elif fc==3:
            self.classifier = nn.Sequential(
                nn.Linear(2304, 256),
                nn.Dropout(d),
                nn.Linear(256, 128),
                nn.Dropout(d),
                nn.Linear(128, 2))
        elif fc==2:
            self.classifier = nn.Sequential(
                nn.Linear(2304, 256),
                nn.Dropout(d),
                nn.Linear(256, 2))
        elif fc==1:
            self.classifier = nn.Sequential(
                nn.Linear(2176, 2))

    def forward(self,dvimg,zsimg,limg,bz,device):
        
        dvzs=self.dvzsmodel
        l=self.lmodel

        dvimg = dvzs.dvmodel.model.conv1(dvimg)
        dvimg = dvzs.dvmodel.model.bn1(dvimg)
        dvimg = dvzs.dvmodel.model.relu(dvimg)
        dvimg = dvzs.dvmodel.model.maxpool(dvimg)
        dvimg = dvzs.dvmodel.model.layer1(dvimg)
        dvimg = dvzs.dvmodel.model.layer2(dvimg)
        dvimg = dvzs.dvmodel.model.layer3(dvimg)
        dvimg = dvzs.dvmodel.model.layer4(dvimg)
        dvpooled_features = self.pooling_layer(dvimg)
        dvpooled_features = dvpooled_features.view(dvpooled_features.size(0), -1)
        dvoutput=dvzs.dvmodel.classifier(dvpooled_features)

        zsimg = dvzs.zsmodel.model.conv1(zsimg)
        zsimg = dvzs.zsmodel.model.bn1(zsimg)
        zsimg = dvzs.zsmodel.model.relu(zsimg)
        zsimg = dvzs.zsmodel.model.maxpool(zsimg)
        zsimg = dvzs.zsmodel.model.layer1(zsimg)
        zsimg = dvzs.zsmodel.model.layer2(zsimg)
        zsimg = dvzs.zsmodel.model.layer3(zsimg)
        zsimg = dvzs.zsmodel.model.layer4(zsimg)
        zspooled_features = self.pooling_layer(zsimg)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)
        zsoutput=dvzs.zsmodel.classifier(zspooled_features)

        img = torch.cat([dvimg, zsimg], dim=1)
        img = dvzs.conv(img)
        imgpooled_features = self.pooling_layer(img)
        dvzspooled_features = imgpooled_features.view(imgpooled_features.size(0), -1)
        dvzsoutput = dvzs.classifier(dvzspooled_features)

        if self.padding==0:
            limg = l.model.conv1(limg)
            limg = l.model.bn1(limg)
            limg = l.model.relu(limg)
            limg = l.model.maxpool(limg)
            limg = l.model.layer1(limg)
            limg = l.model.layer2(limg)
            limg = l.model.layer3(limg)
            limg = l.model.layer4(limg)
            lpooled_features = self.pooling_layer(limg)
            lpooled_features = lpooled_features.view(lpooled_features.size(0), -1)
            loutput = l.classifer(lpooled_features)
        else:
            x = l.conv1(limg)
            x = l.bn1(x)
            x = l.relu(x)
            x = l.maxpool(x)
            x = l.layer1(x)
            x = l.layer2(x)
            x = l.layer3(x)
            x = l.layer4(x)
            feature1 = torch.zeros((x.shape[0], x.shape[1], 1, x.shape[3])).cuda()
            for i in range(x.shape[0]):
                pooling = nn.AdaptiveAvgPool2d((1, int(x.shape[3] * bz[i])))
                feature1[i, :, :, :int(x.shape[3] * bz[i])] = pooling(x[i, :, :, :int(x.shape[3] * bz[i])])


            # 修改wzl 2024-3-18
            # pooled_features = torch.squeeze(feature1)
            pooled_features = torch.squeeze(feature1, dim=2)

            # pooled_features=pooled_features.unsqueeze(0)  #画图用的 记得删掉 
            x1 = l.conv1d(pooled_features)
            lpooled_features=torch.zeros((x1.shape[0],x1.shape[1])).cuda()
            for i in range(x1.shape[0]):
                #原来的
                # feature2 = l.pooling_layer1(x1[i, :, :int(x1.shape[2] * bz[i])])
                # lpooled_features[i]=torch.squeeze(feature2)
                sample = x1[i, :, :int(x1.shape[2] * bz[i])].unsqueeze(0)  # 增加batch_size维度
                feature2 = l.pooling_layer1(sample)
                # feature2 = self.pooling_layer1(x1[i, :, :int(x1.shape[2] * bz[i])]) #原来的
                # 使用 .squeeze() 方法移除所有大小为1的维度，这会将形状从 [1, 2048, 1] 变为 [1, 2048]
                lpooled_features[i] = feature2.squeeze()
                # 现在 feature2_squeezed 的形状是 [1, 2048]，可以被全连接层接受
     

        pooled_features=torch.cat([dvzspooled_features, lpooled_features], dim=1)
        output = self.classifier(pooled_features)
        return output

class ldvzsfusionfccon(nn.Module):
    def __init__(self,dvpath,zspath,lpath,update):
        super().__init__()
        self.dvmodel=resnet(True)
        self.zsmodel=resnet(True)
        self.lmodel=lresnet()
        self.dvmodel.load_state_dict(torch.load(dvpath))
        self.zsmodel.load_state_dict(torch.load(zspath))
        self.lmodel.load_state_dict(torch.load(lpath))
        if update==False:
            for name,module in self.dvmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.zsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
            # for name, module in self.lmodel._modules.items():
            #     for parma in module.parameters():
            #         parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            # nn.Linear(256, 128),
            # nn.Dropout(0.5),
            nn.Linear(256, 2))

    def forward(self,dvimg,zsimg,limg):
        dv=self.dvmodel
        zs=self.zsmodel
        l=self.lmodel

        dvimg = dv.model.conv1(dvimg)
        dvimg = dv.model.bn1(dvimg)
        dvimg = dv.model.relu(dvimg)
        dvimg = dv.model.maxpool(dvimg)
        dvimg = dv.model.layer1(dvimg)
        dvimg = dv.model.layer2(dvimg)
        dvimg = dv.model.layer3(dvimg)
        dvimg = dv.model.layer4(dvimg)
        dvpooled_features = self.pooling_layer(dvimg)
        dvpooled_features = dvpooled_features.view(dvpooled_features.size(0), -1)
        dvoutput=dv.classifier(dvpooled_features)

        zsimg = zs.model.conv1(zsimg)
        zsimg = zs.model.bn1(zsimg)
        zsimg = zs.model.relu(zsimg)
        zsimg = zs.model.maxpool(zsimg)
        zsimg = zs.model.layer1(zsimg)
        zsimg = zs.model.layer2(zsimg)
        zsimg = zs.model.layer3(zsimg)
        zsimg = zs.model.layer4(zsimg)
        zspooled_features = self.pooling_layer(zsimg)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)
        zsoutput=zs.classifier(zspooled_features)

        limg = l.model.conv1(limg)
        limg = l.model.bn1(limg)
        limg = l.model.relu(limg)
        limg = l.model.maxpool(limg)
        limg = l.model.layer1(limg)
        limg = l.model.layer2(limg)
        limg = l.model.layer3(limg)
        limg = l.model.layer4(limg)
        lpooled_features = self.pooling_layer(limg)
        lpooled_features = lpooled_features.view(lpooled_features.size(0), -1)
        loutput=l.classifer(lpooled_features)

        pooled_features=torch.cat([dvpooled_features, zspooled_features,lpooled_features], dim=1)
        output = self.classifier(pooled_features)
        return output,loutput,dvoutput

class dvzsfusionfcadd(nn.Module):
    def __init__(self,dvpath,zspath,lpath,update):
        super().__init__()
        self.dvmodel=resnet(True)
        self.zsmodel=resnet(True)
        self.dvmodel.load_state_dict(torch.load(dvpath))
        self.zsmodel.load_state_dict(torch.load(zspath))
        if update==False:
            for name,module in self.dvmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.zsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            # nn.Dropout(0.3),
            # nn.Linear(1024, 512),
            # nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.Dropout(0.2),
            nn.Linear(128, 2))

    def forward(self,dvimg,zsimg,limg):
        dv=self.dvmodel
        zs=self.zsmodel

        dvimg = dv.model.conv1(dvimg)
        dvimg = dv.model.bn1(dvimg)
        dvimg = dv.model.relu(dvimg)
        dvimg = dv.model.maxpool(dvimg)
        dvimg = dv.model.layer1(dvimg)
        dvimg = dv.model.layer2(dvimg)
        dvimg = dv.model.layer3(dvimg)
        dvimg = dv.model.layer4(dvimg)
        dvpooled_features = self.pooling_layer(dvimg)
        dvpooled_features = dvpooled_features.view(dvpooled_features.size(0), -1)

        zsimg = zs.model.conv1(zsimg)
        zsimg = zs.model.bn1(zsimg)
        zsimg = zs.model.relu(zsimg)
        zsimg = zs.model.maxpool(zsimg)
        zsimg = zs.model.layer1(zsimg)
        zsimg = zs.model.layer2(zsimg)
        zsimg = zs.model.layer3(zsimg)
        zsimg = zs.model.layer4(zsimg)
        zspooled_features = self.pooling_layer(zsimg)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)

        pooled_features=dvpooled_features+zspooled_features
        output = self.classifier(pooled_features)
        return output

class ldvzsfusionfcadd(nn.Module):
    def __init__(self,dvpath,zspath,lpath,update):
        super().__init__()
        self.dvmodel=resnet(True)
        self.zsmodel=resnet(True)
        self.lmodel=lresnet()
        self.dvmodel.load_state_dict(torch.load(dvpath))
        self.zsmodel.load_state_dict(torch.load(zspath))
        self.lmodel.load_state_dict(torch.load(lpath))
        if update==False:
            for name,module in self.dvmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.zsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
            for name, module in self.lmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            # nn.Dropout(0.3),
            # nn.Linear(1536, 512),
            # nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.Dropout(0.2),
            nn.Linear(128, 2))

    def forward(self,dvimg,zsimg,limg):
        dv=self.dvmodel
        zs=self.zsmodel
        l=self.lmodel

        dvimg = dv.model.conv1(dvimg)
        dvimg = dv.model.bn1(dvimg)
        dvimg = dv.model.relu(dvimg)
        dvimg = dv.model.maxpool(dvimg)
        dvimg = dv.model.layer1(dvimg)
        dvimg = dv.model.layer2(dvimg)
        dvimg = dv.model.layer3(dvimg)
        dvimg = dv.model.layer4(dvimg)
        dvpooled_features = self.pooling_layer(dvimg)
        dvpooled_features = dvpooled_features.view(dvpooled_features.size(0), -1)

        zsimg = zs.model.conv1(zsimg)
        zsimg = zs.model.bn1(zsimg)
        zsimg = zs.model.relu(zsimg)
        zsimg = zs.model.maxpool(zsimg)
        zsimg = zs.model.layer1(zsimg)
        zsimg = zs.model.layer2(zsimg)
        zsimg = zs.model.layer3(zsimg)
        zsimg = zs.model.layer4(zsimg)
        zspooled_features = self.pooling_layer(zsimg)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)

        limg = l.model.conv1(limg)
        limg = l.model.bn1(limg)
        limg = l.model.relu(limg)
        limg = l.model.maxpool(limg)
        limg = l.model.layer1(limg)
        limg = l.model.layer2(limg)
        limg = l.model.layer3(limg)
        limg = l.model.layer4(limg)
        lpooled_features = self.pooling_layer(limg)
        lpooled_features = lpooled_features.view(lpooled_features.size(0), -1)

        pooled_features=dvpooled_features+zspooled_features+lpooled_features
        output = self.classifier(pooled_features)
        return output

class dvzsfusionconvcon(nn.Module):
    def __init__(self,dvpath,zspath,lpath,update):
        super().__init__()
        self.dvmodel=resnet(True)
        self.zsmodel=resnet(True)
        self.dvmodel.load_state_dict(torch.load(dvpath))
        self.zsmodel.load_state_dict(torch.load(zspath))
        if update==False:
            for name,module in self.dvmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.zsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 2))

    def forward(self,dvimg,zsimg,limg):
        dv=self.dvmodel
        zs=self.zsmodel

        dvimg = dv.model.conv1(dvimg)
        dvimg = dv.model.bn1(dvimg)
        dvimg = dv.model.relu(dvimg)
        dvimg = dv.model.maxpool(dvimg)
        dvimg = dv.model.layer1(dvimg)
        dvimg = dv.model.layer2(dvimg)
        dvimg = dv.model.layer3(dvimg)
        dvimg = dv.model.layer4(dvimg)

        zsimg = zs.model.conv1(zsimg)
        zsimg = zs.model.bn1(zsimg)
        zsimg = zs.model.relu(zsimg)
        zsimg = zs.model.maxpool(zsimg)
        zsimg = zs.model.layer1(zsimg)
        zsimg = zs.model.layer2(zsimg)
        zsimg = zs.model.layer3(zsimg)
        zsimg = zs.model.layer4(zsimg)

        img=torch.cat([dvimg,zsimg],dim=1)
        zspooled_features = self.pooling_layer(img)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)
        output = self.classifier(zspooled_features)
        return output

class dvzsfusionconvadd(nn.Module):
    def __init__(self,dvpath,zspath,lpath,update):
        super().__init__()
        self.dvmodel=resnet(True)
        self.zsmodel=resnet(True)
        self.dvmodel.load_state_dict(torch.load(dvpath))
        self.zsmodel.load_state_dict(torch.load(zspath))
        if update==False:
            for name,module in self.dvmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.zsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            # nn.Dropout(0.3),
            # nn.Linear(1024, 512),
            # nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.Dropout(0.2),
            nn.Linear(128, 2))

    def forward(self,dvimg,zsimg,limg):
        dv=self.dvmodel
        zs=self.zsmodel

        dvimg = dv.model.conv1(dvimg)
        dvimg = dv.model.bn1(dvimg)
        dvimg = dv.model.relu(dvimg)
        dvimg = dv.model.maxpool(dvimg)
        dvimg = dv.model.layer1(dvimg)
        dvimg = dv.model.layer2(dvimg)
        dvimg = dv.model.layer3(dvimg)
        dvimg = dv.model.layer4(dvimg)

        zsimg = zs.model.conv1(zsimg)
        zsimg = zs.model.bn1(zsimg)
        zsimg = zs.model.relu(zsimg)
        zsimg = zs.model.maxpool(zsimg)
        zsimg = zs.model.layer1(zsimg)
        zsimg = zs.model.layer2(zsimg)
        zsimg = zs.model.layer3(zsimg)
        zsimg = zs.model.layer4(zsimg)

        img=dvimg+zsimg
        zspooled_features = self.pooling_layer(img)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)
        output = self.classifier(zspooled_features)
        return output

class dvzsfusionconvcon_con(nn.Module):
    def __init__(self,dvpath,zspath,lpath,update,d):
        super().__init__()
        self.dvmodel=resnet(True)
        self.zsmodel=resnet(True)
        self.dvmodel.load_state_dict(torch.load(dvpath))
        self.zsmodel.load_state_dict(torch.load(zspath))
        if update==False:
            for name,module in self.dvmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.zsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Sequential(nn.Conv2d(1024,512, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                # nn.ReLU(inplace=True),
                                # nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                # nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                )
        self.classifier = nn.Sequential(
            # nn.Linear(512, 256),
            # nn.Dropout(d),
            nn.Linear(256, 128),
            nn.Dropout(d),
            nn.Linear(128, 64),
            nn.Dropout(d),
            # nn.Linear(64, 32),
            # nn.Dropout(d),
            nn.Linear(64, 2))

    def forward(self,dvimg,zsimg,limg,bz,device):
        dv=self.dvmodel
        zs=self.zsmodel

        dvimg = dv.model.conv1(dvimg)
        dvimg = dv.model.bn1(dvimg)
        dvimg = dv.model.relu(dvimg)
        dvimg = dv.model.maxpool(dvimg)
        dvimg = dv.model.layer1(dvimg)
        dvimg = dv.model.layer2(dvimg)
        dvimg = dv.model.layer3(dvimg)
        dvimg = dv.model.layer4(dvimg)
        dvpooled_features = self.pooling_layer(dvimg)
        dvpooled_features = dvpooled_features.view(dvpooled_features.size(0), -1)
        dvoutput = dv.classifier(dvpooled_features)

        zsimg = zs.model.conv1(zsimg)
        zsimg = zs.model.bn1(zsimg)
        zsimg = zs.model.relu(zsimg)
        zsimg = zs.model.maxpool(zsimg)
        zsimg = zs.model.layer1(zsimg)
        zsimg = zs.model.layer2(zsimg)
        zsimg = zs.model.layer3(zsimg)
        zsimg = zs.model.layer4(zsimg)
        zspooled_features = self.pooling_layer(zsimg)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)
        zsoutput = zs.classifier(zspooled_features)

        img=torch.cat([dvimg,zsimg],dim=1)
        img=self.conv(img)
        imgpooled_features = self.pooling_layer(img)
        imgpooled_features = imgpooled_features.view(imgpooled_features.size(0), -1)
        output = self.classifier(imgpooled_features)
        return output

class ldvzsconvconconv_fccon(nn.Module):
    def __init__(self,dvpath,zspath,lpath,update):
        super().__init__()
        self.dvmodel=resnet(True)
        self.zsmodel=resnet(True)
        self.lmodel=lresnet()
        self.dvmodel.load_state_dict(torch.load(dvpath))
        self.zsmodel.load_state_dict(torch.load(zspath))
        self.lmodel.load_state_dict(torch.load(lpath))
        if update==False:
            for name,module in self.dvmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.zsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                # nn.ReLU(inplace=True),
                                # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                # nn.ReLU(inplace=True),
                                # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 2))

    def forward(self,dvimg,zsimg,limg):
        dv=self.dvmodel
        zs=self.zsmodel
        l=self.lmodel

        dvimg = dv.model.conv1(dvimg)
        dvimg = dv.model.bn1(dvimg)
        dvimg = dv.model.relu(dvimg)
        dvimg = dv.model.maxpool(dvimg)
        dvimg = dv.model.layer1(dvimg)
        dvimg = dv.model.layer2(dvimg)
        dvimg = dv.model.layer3(dvimg)
        dvimg = dv.model.layer4(dvimg)
        dvpooled_features = self.pooling_layer(dvimg)
        dvpooled_features = dvpooled_features.view(dvpooled_features.size(0), -1)
        dvoutput = dv.classifier(dvpooled_features)

        zsimg = zs.model.conv1(zsimg)
        zsimg = zs.model.bn1(zsimg)
        zsimg = zs.model.relu(zsimg)
        zsimg = zs.model.maxpool(zsimg)
        zsimg = zs.model.layer1(zsimg)
        zsimg = zs.model.layer2(zsimg)
        zsimg = zs.model.layer3(zsimg)
        zsimg = zs.model.layer4(zsimg)
        zspooled_features = self.pooling_layer(zsimg)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)
        zsoutput = zs.classifier(zspooled_features)

        dvzs=torch.cat([dvimg,zsimg],dim=1)
        dvzs=self.conv(dvzs)
        dvzspooled_features = self.pooling_layer(dvzs)
        dvzspooled_features = dvzspooled_features.view(dvzspooled_features.size(0), -1)

        limg = l.model.conv1(limg)
        limg = l.model.bn1(limg)
        limg = l.model.relu(limg)
        limg = l.model.maxpool(limg)
        limg = l.model.layer1(limg)
        limg = l.model.layer2(limg)
        limg = l.model.layer3(limg)
        limg = l.model.layer4(limg)
        lpooled_features = self.pooling_layer(limg)
        lpooled_features = lpooled_features.view(lpooled_features.size(0), -1)
        loutput=l.classifer(lpooled_features)

        img=torch.cat([dvzspooled_features,lpooled_features],dim=1)
        output = self.classifier(img)
        return output,loutput,dvoutput

class dvzsfusionconvcon_con_con(nn.Module):
    def __init__(self,dvpath,zspath,lpath,update):
        super().__init__()
        self.dvmodel=resnet(True)
        self.zsmodel=resnet(True)
        self.dvmodel.load_state_dict(torch.load(dvpath))
        self.zsmodel.load_state_dict(torch.load(zspath))
        if update==False:
            for name,module in self.dvmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.zsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                  nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),)
        # self.conv2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False),
        #                            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True,
        #                                           track_running_stats=True), )
        self.conv=nn.Sequential(nn.Conv2d(640, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                # nn.ReLU(inplace=True),
                                # nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                # nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                )
        self.classifier = nn.Sequential(
            # nn.Linear(512, 256),
            # nn.Dropout(0.5),
            # nn.Linear(256, 128),
            # nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 2))

    def forward(self,dvimg,zsimg,limg):
        dv=self.dvmodel
        zs=self.zsmodel

        dvimg = dv.model.conv1(dvimg)
        dvimg = dv.model.bn1(dvimg)
        dvimg = dv.model.relu(dvimg)
        dvimg = dv.model.maxpool(dvimg)
        dvimg = dv.model.layer1(dvimg)
        dvimg = dv.model.layer2(dvimg)
        dvimg = dv.model.layer3(dvimg)
        dvimg = dv.model.layer4(dvimg)
        # dvimg=self.conv1(dvimg)

        zsimg = zs.model.conv1(zsimg)
        zsimg = zs.model.bn1(zsimg)
        zsimg = zs.model.relu(zsimg)
        zsimg = zs.model.maxpool(zsimg)
        zsimg = zs.model.layer1(zsimg)
        zsimg = zs.model.layer2(zsimg)
        zsimg = zs.model.layer3(zsimg)
        zsimg = zs.model.layer4(zsimg)
        zsimg=self.conv1(zsimg)

        img=torch.cat([dvimg,zsimg],dim=1)
        img=self.conv(img)
        zspooled_features = self.pooling_layer(img)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)
        output = self.classifier(zspooled_features)
        return output

class dvzsattentionfusion(nn.Module):
    def __init__(self,dvpath,zspath,lpath,update,fway):
        super().__init__()
        self.dvmodel=resnet(True)
        self.zsmodel=resnet(True)
        self.dvmodel.load_state_dict(torch.load(dvpath))
        self.zsmodel.load_state_dict(torch.load(zspath))
        self.fway=fway
        if update==False:
            for name,module in self.dvmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad=False
            for name, module in self.zsmodel._modules.items():
                for parma in module.parameters():
                    parma.requires_grad = False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Sequential(nn.Conv2d(1024, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                # nn.ReLU(inplace=True),
                                # nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                # nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                )
        if self.fway == 'con3':
            self.attentionconv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                  nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                  nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                  )
        elif self.fway == 'con4':
            self.attentionconv = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                  nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  # nn.ReLU(inplace=True),
                                  # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                  # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                  )
        elif self.fway=='add':
            self.attentionconv = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
        self.classifier = nn.Sequential(
            # nn.Linear(256, 128),
            # nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )

    def forward(self,dvimg,zsimg,limg):
        dv=self.dvmodel
        zs=self.zsmodel

        dvimg = dv.model.conv1(dvimg)
        dvimg = dv.model.bn1(dvimg)
        dvimg = dv.model.relu(dvimg)
        dvimg = dv.model.maxpool(dvimg)
        dvimg = dv.model.layer1(dvimg)
        dvimg = dv.model.layer2(dvimg)
        dvimg = dv.model.layer3(dvimg)
        dvimg = dv.model.layer4(dvimg)
        # dvimg1 = dv.model.layer4(dvimg)
        # dvpooled_features = self.pooling_layer(dvimg1)
        # dvpooled_features = dvpooled_features.view(dvpooled_features.size(0), -1)
        # dvoutput = dv.classifier(dvpooled_features)

        zsimg = zs.model.conv1(zsimg)
        zsimg = zs.model.bn1(zsimg)
        zsimg = zs.model.relu(zsimg)
        zsimg = zs.model.maxpool(zsimg)
        zsimg = zs.model.layer1(zsimg)
        zsimg = zs.model.layer2(zsimg)
        zsimg = zs.model.layer3(zsimg)
        zsimg = zs.model.layer4(zsimg)
        # zsimg1 = zs.model.layer4(zsimg)
        # zspooled_features = self.pooling_layer(zsimg1)
        # zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)
        # zsoutput = zs.classifier(zspooled_features)

        zsimg = self.attentionconv(zsimg)
        if self.fway in ['con3','con4'] :
            zsimg = torch.mean(zsimg, dim=1, keepdim=True)
            zsimg = zsimg.repeat(1, dvimg.shape[1], 1, 1)
            zsimg = zsimg * dvimg
            img=torch.cat([dvimg,zsimg],dim=1)
            img = self.conv(img)
        elif self.fway=='add':
            img=dvimg+zsimg
        zspooled_features = self.pooling_layer(img)
        zspooled_features = zspooled_features.view(zspooled_features.size(0), -1)
        output = self.classifier(zspooled_features)
        return output
