import cv2
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import random
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor

class mobv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=models.mobilenet_v2(pretrained=True) #使用预训练好的模型参数
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.model.classifier = nn.Sequential(
            # nn.Linear(1280, 512),
            # nn.Dropout(0.5),
            # nn.Linear(512,128),
            # nn.Dropout(0.5),
            nn.Linear(1280,2))
        # self.model.classifier = nn.Sequential(
        #     nn.Dropout(0.8),
        #     nn.Linear(1280, 2))
        # self.model.classifier[2]=nn.Dropout(0.7)
        # self.model.classifier[3]=nn.Sequential(
        #     nn.Linear(1280, 128),
        #     nn.Dropout(0.7),
        #     nn.Linear(128,2))

    def forward(self,x):
        # n=self.model
        # x = n.conv1(x)
        # x = n.bn1(x)
        # x = n.relu(x)
        # x = n.maxpool(x)
        # x = n.layer1(x)
        # x = n.layer2(x)
        # x = n.layer3(x)
        # x = n.layer4(x)
        # pooled_features = self.pooling_layer(x)
        # # pooled_features=self.dropout(pooled_features)
        #
        # pooled_features = pooled_features.view(pooled_features.size(0), -1)
        # output = self.classifer(pooled_features)
        output=self.model(x)
        return output

class resnet(nn.Module):
    def __init__(self,pre):
        super().__init__()
        self.model=models.resnet18(pretrained=pre) #使用预训练好的模型参数
        # for name,module in self.model._modules.items():
        #     for parma in module.parameters():
        #         parma.requires_grad=False
        # for name,module in self.model._modules.items():
        #     if name=='layer4':
        #         for parma in module.parameters():
        #             parma.requires_grad=True
        #     else:
        #         for parma in module.parameters():
        #             parma.requires_grad=False
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        # self.model.fc=nn.Linear(512,2)
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(d),
        #     nn.Linear(512, 2))
        self.classifier = nn.Sequential(
            # nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.Dropout(0.4),
            nn.Linear(128,2))
        # self.model.fc=nn.Linear(512,2)
        # self.model.classifier[2]=nn.Dropout(0.7)
        # self.model.classifier[3]=nn.Sequential(
        #     nn.Linear(1280, 128),
        #     nn.Dropout(0.7),
        #     nn.Linear(128,2))

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
        # pooled_features=self.dropout(pooled_features)

        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifier(pooled_features)
        # output=self.model(x)
        return output

class resnetsa(nn.Module):
    def __init__(self,pre):
        super().__init__()
        self.model=models.resnet18(pretrained=pre) #使用预训练好的模型参数
        self.samodel=models.resnet18(pretrained=True)
        # for name,module in self.samodel._modules.items():
        #     for parma in module.parameters():
        #         parma.requires_grad=False
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            # nn.Dropout(0.3),
            nn.Linear(1024,256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.Dropout(0.2),
            nn.Linear(128,2))

    def forward(self,x,xsa):
        n=self.model
        s=self.samodel
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        x = n.layer4(x)

        xsa = s.conv1(xsa)
        xsa = s.bn1(xsa)
        xsa = s.relu(xsa)
        xsa= s.maxpool(xsa)
        xsa = s.layer1(xsa)
        xsa = s.layer2(xsa)
        xsa = s.layer3(xsa)
        xsa = s.layer4(xsa)

        img = torch.concat([x, xsa], dim=1)
        pooled_features = self.pooling_layer(img)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifier(pooled_features)
        return output

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
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
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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

class ResNet(nn.Module):
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer( 64, layers[0])
        self.layer2 = self._make_layer( 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer( 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(256 * BasicBlock.expansion, 128),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.Dropout(0.2),
            # nn.Linear(64, 32),
            # nn.Dropout(0.2),
            nn.Linear(64,num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        # block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                norm_layer(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(
            BasicBlock(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class resnet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=models.resnet18(pretrained=True) #使用预训练好的模型参数
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv4=nn.Conv2d(256,512,3,2,1,bias=False)
        # self.bn4=nn.BatchNorm2d(512)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128,64),
            nn.Dropout(0.5),
            nn.Linear(64,2))

    def forward(self,x,bz,device):
        n=self.model
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        # x = self.conv4(x)
        # x=self.bn4(x)
        output=torch.zeros((x.shape[0],2)).cuda()
        for i in range(x.shape[0]):
            feature1=self.pooling_layer(x[i,:,:,:int(x.shape[3] * bz[i])])
            output[i]=self.classifer(feature1.view(feature1.shape[0]))
        return output

class resnet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=models.resnet18(pretrained=True) #使用预训练好的模型参数
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv4=nn.Conv2d(256,512,3,2,1,bias=False)
        self.bn4=nn.BatchNorm2d(512)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(512, 2)

    def forward(self,x,bz,device):
        n=self.model
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        x = self.conv4(x)
        x=self.bn4(x)
        output=torch.zeros((x.shape[0],2)).cuda()
        for i in range(x.shape[0]):
            feature1=self.pooling_layer(x[i,:,:,:int(x.shape[3] * bz[i])])
            output[i]=self.classifer(feature1.view(feature1.shape[0]))
        return output

class resnet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=models.resnet18(pretrained=True) #使用预训练好的模型参数
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(512, 2)
        for name,module in self.model._modules.items():
            if name=='layer4':
                for parma in module.parameters():
                    parma.requires_grad=True
            else:
                for parma in module.parameters():
                    parma.requires_grad=False

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
        output=torch.zeros((x.shape[0],2)).cuda()
        for i in range(x.shape[0]):
            print(x[i,:,:,:int(x.shape[3] * bz[i])].shape)
            feature1=self.pooling_layer(x[i,:,:,:int(x.shape[3] * bz[i])])
            output[i]=self.classifer(feature1.view(feature1.shape[0]))
        return output

class resnetpre(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=models.resnet18(pretrained=False) #使用预训练好的模型参数
        self.model.fc=nn.Linear(512,10)
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
    def forward(self,x):
        n=self.model
        output=n(x)
        return output

class trainresnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=resnetpre() #使用预训练好的模型参数
        self.model=torch.load('../../trainresnet/trainresnet/epoch_44.pth')
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x,bz,device):
        n=self.model.model
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        x = n.layer4(x)
        output = torch.zeros((x.shape[0], 2)).cuda()
        for i in range(x.shape[0]):
            feature1 = self.pooling_layer(x[i, :, :, :int(x.shape[3] * bz[i])])
            output[i] = self.classifer(feature1.view(feature1.shape[0]))
        return output

class trainresnet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=resnetpre() #使用预训练好的模型参数
        self.model=torch.load('../../trainresnet/trainresnet/epoch_44.pth')
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        # self.conv4 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        # self.bn4 = nn.BatchNorm2d(512)
        self.classifer = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x,bz,device):
        n=self.model.model
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        # x = self.conv4(x)
        # x = self.bn4(x)
        output = torch.zeros((x.shape[0], 2)).cuda()
        for i in range(x.shape[0]):
            feature1 = self.pooling_layer(x[i, :, :, :int(x.shape[3] * bz[i])])
            output[i] = self.classifer(feature1.view(feature1.shape[0]))
        return output

class trainresnet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=resnetpre() #使用预训练好的模型参数
        self.model=torch.load('../../trainresnet/trainresnet/epoch_44.pth')
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.classifer = nn.Linear(512, 2)
        # self.dropout = nn.Dropout(0.5)

    def forward(self,x,bz,device):
        n=self.model.model
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        output = torch.zeros((x.shape[0], 2)).cuda()
        for i in range(x.shape[0]):
            feature1 = self.pooling_layer(x[i, :, :, :int(x.shape[3] * bz[i])])
            output[i] = self.classifer(feature1.view(feature1.shape[0]))
        return output

class trainresnet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=resnetpre() #使用预训练好的模型参数
        self.model=torch.load('../../trainresnet/trainresnet/epoch_44.pth')
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(512, 2)
        for name,module in self.model.model._modules.items():
            if name=='layer4':
                for parma in module.parameters():
                    parma.requires_grad=True
            else:
                for parma in module.parameters():
                    parma.requires_grad=False
    def forward(self,x,bz,device):
        n=self.model.model
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        x = n.layer4(x)
        output = torch.zeros((x.shape[0], 2)).cuda()
        for i in range(x.shape[0]):
            feature1 = self.pooling_layer(x[i, :, :, :int(x.shape[3] * bz[i])])
            output[i] = self.classifer(feature1.view(feature1.shape[0]))
        return output

class mynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Sequential(
            nn.Linear(512,256),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.Dropout(0.5),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.model(x)
        pooled_features = self.pooling_layer(x)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifier(pooled_features)
        return output

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(3407)
#
# mo=models.resnet18(False)
# print(mo._modules.items())

# mo=resnet2()
# if torch.cuda.is_available():
#     device = torch.device(f'cuda:1')
#     print('use gpu')
# else:
#     device = torch.device('cpu')
#     print('use cpu')
# img=torch.zeros((1,1,50,14300)).to(device)
# bz=torch.tensor([0.5]).to(device)
#
# mo.to(device)
#
# img1=mo.forward(img,bz,device)
# print()

# img0=cv2.imread('0.jpg')
# img1=cv2.imread('1.jpg')
# print((img0==img1).all())