import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import random
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor

class resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=models.resnet18(pretrained=True) #使用预训练好的模型参数
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        # self.classifer = nn.Linear(512, 2)
        self.classifer = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            # nn.Linear(256,128),
            # nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.Dropout(0.5),
            nn.Linear(256, 2))
        # self.dropout=nn.Dropout(0.8)

    def forward(self,x,bz,device):
        n=self.model
        x = n.conv1(x)
        #x = n.bn1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        # x=self.dropout(x)
        x = n.layer2(x)
        # x = self.dropout(x)
        x = n.layer3(x)
        # x = self.dropout(x)
        x = n.layer4(x)
        # x = self.dropout(x)
        # output=torch.zeros((x.shape[0],2)).cuda()
        # for i in range(x.shape[0]):
        #     feature1=self.pooling_layer(x[i,:,:,:int(x.shape[3] * bz[i])])
        #     output[i]=self.classifer(feature1.view(feature1.shape[0]))

        pooled_features = self.pooling_layer(x)
        # pooled_features=self.dropout(pooled_features)

        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifer(pooled_features)
        return output

class resnetconv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=models.resnet18(pretrained=True) #使用预训练好的模型参数
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pooling_layer = nn.AdaptiveAvgPool2d((1,32))
        self.pooling_layer1 = nn.AdaptiveAvgPool1d(1)
        # self.classifer = nn.Linear(512, 2)
        self.classifer = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.Dropout(0.5),
            nn.Linear(128, 2))
        # self.dropout=nn.Dropout(0.8)
        self.conv1d=nn.Sequential(
            nn.Conv1d(512,1024,3,2,1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024,2048,5,2,1,bias=False)
        )

    def forward(self,x,bz,device):
        n=self.model
        x = n.conv1(x)
        #x = n.bn1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        # x=self.dropout(x)
        x = n.layer2(x)
        # x = self.dropout(x)
        x = n.layer3(x)
        # x = self.dropout(x)
        x = n.layer4(x)
        # x = self.dropout(x)
        # output=torch.zeros((x.shape[0],2)).cuda()
        # for i in range(x.shape[0]):
        #     feature1=self.pooling_layer(x[i,:,:,:int(x.shape[3] * bz[i])])
        #     output[i]=self.classifer(feature1.view(feature1.shape[0]))

        pooled_features = self.pooling_layer(x)
        # pooled_features=self.dropout(pooled_features)

        pooled_features = torch.squeeze(pooled_features)
        x1=self.conv1d(pooled_features)
        x1=self.pooling_layer1(x1)
        x1 = x1.view(x1.size(0), -1)
        output = self.classifer(x1)
        return output

class resnetpadding(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=models.resnet18(pretrained=True) #使用预训练好的模型参数
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv4=nn.Conv2d(256,512,3,2,1,bias=False)
        # self.bn4=nn.BatchNorm2d(512)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128,2))

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
        # x = self.conv4(x)
        # x=self.bn4(x)
        output=torch.zeros((x.shape[0],2)).cuda()
        for i in range(x.shape[0]):
            feature1=self.pooling_layer(x[i,:,:,:int(x.shape[3] * bz[i])])
            output[i]=self.classifer(feature1.view(feature1.shape[0]))
        return output

class resnetconv1dpadding(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=models.resnet18(pretrained=True) #使用预训练好的模型参数
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pooling_layer = nn.AdaptiveAvgPool2d((1,32))
        self.pooling_layer1 = nn.AdaptiveAvgPool1d(1)
        # self.classifer = nn.Linear(512, 2)
        self.classifer = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Dropout(0.5),
            # nn.Linear(256,128),
            # nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.Dropout(0.5),
            nn.Linear(256, 2))
        # self.dropout=nn.Dropout(0.8)
        self.conv1d=nn.Sequential(
            nn.Conv1d(512,1024,5,2,1,bias=False),
            nn.BatchNorm1d(1024,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024,2048,3,1,1,bias=False),
            # nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(2048,512,3,1,1,bias=False)
        )

    def forward(self,x,bz,device):
        n=self.model
        x = n.conv1(x)
        #x = n.bn1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        # x=self.dropout(x)
        x = n.layer2(x)
        # x = self.dropout(x)
        x = n.layer3(x)
        # x = self.dropout(x)
        x = n.layer4(x)
        # x = self.dropout(x)
        feature1=torch.zeros((x.shape[0],x.shape[1],1,32)).cuda()
        for i in range(x.shape[0]):
            feature1[i,:,:,:int(x.shape[3] * bz[i])]=self.pooling_layer(x[i,:,:,:int(x.shape[3] * bz[i])])
        #     output[i]=self.classifer(feature1.view(feature1.shape[0]))

        # pooled_features = self.pooling_layer(x)
        # pooled_features=self.dropout(pooled_features)

        pooled_features = torch.squeeze(feature1)
        x1=self.conv1d(pooled_features)
        output=torch.zeros((x1.shape[0],2)).cuda()
        for i in range(x1.shape[0]):
            feature2=self.pooling_layer1(x1[i,:,:int(x1.shape[2] * bz[i])])
            output[i]=self.classifer(feature2.view(feature2.shape[0]))
        return output

class resnetconv1drpadding(nn.Module):
    def __init__(self,conv1,conv2):
        super().__init__()
        self.model=models.resnet18(pretrained=True) #使用预训练好的模型参数
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.pooling_layer = nn.AdaptiveAvgPool2d((1,32))
        self.pooling_layer1 = nn.AdaptiveAvgPool1d(1)
        # self.classifer = nn.Linear(512, 2)
        self.classifer = nn.Sequential(
            nn.Linear(conv2, 256),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.Dropout(0.5),
            nn.Linear(128, 2))
        # self.dropout=nn.Dropout(0.8)
        self.conv1d=nn.Sequential(
            nn.Conv1d(512,conv1,3,2,1,bias=False),
            nn.BatchNorm1d(conv1,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv1,conv2,5,2,1,bias=False),
            nn.BatchNorm1d(conv2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(128,512,3,1,1,bias=False),
            # nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

    def forward(self,x,bz,device):
        n=self.model
        x = n.conv1(x)
        #x = n.bn1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        # x=self.dropout(x)
        x = n.layer2(x)
        # x = self.dropout(x)
        x = n.layer3(x)
        # x = self.dropout(x)
        x = n.layer4(x)
        # x = self.dropout(x)
        feature1=torch.zeros((x.shape[0],x.shape[1],1,x.shape[3])).cuda()
        for i in range(x.shape[0]):
            pooling=nn.AdaptiveAvgPool2d((1,int(x.shape[3] * bz[i])))
            feature1[i,:,:,:int(x.shape[3] * bz[i])]=pooling(x[i,:,:,:int(x.shape[3] * bz[i])])
        pooled_features = torch.squeeze(feature1)
        x1=self.conv1d(pooled_features)
        output=torch.zeros((x1.shape[0],2)).cuda()
        for i in range(x1.shape[0]):
            feature2=self.pooling_layer1(x1[i,:,:int(x1.shape[2] * bz[i])])
            output[i]=self.classifer(feature2.view(feature2.shape[0]))
        return output

def conv3x3(in_planes: int, out_planes: int, kernel_size,padding,stride:int=1, groups: int = 1, dilation: int=1) -> nn.Conv2d:
    """
    创建一个3x3的卷积层，包括填充(padding)。

    参数:
    in_planes (int): 输入通道数。
    out_planes (int): 输出通道数。
    kernel_size: 卷积核的大小，虽然函数名是conv3x3，但可以通过这个参数自定义卷积核大小。
    padding: 卷积层的填充。
    stride (int): 卷积的步长，默认为1。
    groups (int): 分组卷积的组数，默认为1，当设置为2时，输入和输出通道被分成两组，各自进行卷积操作。
    dilation (int): 卷积核元素之间的间距，用于扩张卷积(dilated convolution)，默认为1。

    返回:
    nn.Conv2d: 创建的卷积层。
    """
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
    expansion: int = 1  # 扩展系数，用于调整网络通道数的增长

    def __init__(
        self,
        inplanes: int,  # 输入通道数
        planes: int,  # 输出通道数
        kernel: tuple,  # 卷积核大小
        stride: int = 1,  # 卷积步长
        downsample: Optional[nn.Module] = None,  # 降采样模块，用于调整维度一致性
        groups: int = 1,  # 分组卷积的组数，BasicBlock默认为1
        base_width: int = 64,  # 基础宽度，BasicBlock中未使用，但为了接口一致性保留
        dilation: int = 1,  # 空洞卷积的膨胀率，BasicBlock默认为1
        padding: tuple = (0,0),  # 卷积填充
        norm_layer: Optional[Callable[..., nn.Module]] = None,  # 标准化层，默认为批量归一化
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 若未指定标准化层，则使用BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock只支持groups=1且base_width=64")
        if dilation > 1:
            raise NotImplementedError("BasicBlock不支持dilation > 1")

        # 第一个卷积层，可能进行下采样（如果stride != 1）
        self.conv1 = conv3x3(inplanes, planes, kernel, padding, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)  # 使用原地操作提高效率
        # 第二个卷积层，不改变特征图的尺寸
        self.conv2 = conv3x3(planes, planes, kernel, padding)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample  # 可选的下采样模块
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # 保留输入，用于残差连接

        # 通过第一个卷积->批归一化->ReLU激活函数
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 通过第二个卷积->批归一化
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果有下采样，则对输入进行下采样以匹配维度
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 加上残差
        out = self.relu(out)  # 再次通过ReLU激活函数

        return out


class conv2dconv1dpadding(nn.Module):
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
        self.pooling_layer = nn.AdaptiveAvgPool2d((1, 32))
        self.pooling_layer1 = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
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
            # nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(2048, 512, 3, 1, 1, bias=False)
        )

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
        # x=self.dropout(x)
        x = self.layer2(x)
        # x = self.dropout(x)
        x = self.layer3(x)
        # x = self.dropout(x)
        x = self.layer4(x)
        # x = self.dropout(x)
        feature1 = torch.zeros((x.shape[0], x.shape[1], 1, 32)).cuda()
        for i in range(x.shape[0]):
            feature1[i, :, :, :int(x.shape[3] * bz[i])] = self.pooling_layer(x[i, :, :, :int(x.shape[3] * bz[i])])
        #     output[i]=self.classifer(feature1.view(feature1.shape[0]))

        # pooled_features = self.pooling_layer(x)
        # pooled_features=self.dropout(pooled_features)

        pooled_features = torch.squeeze(feature1)
        x1 = self.conv1d(pooled_features)
        output = torch.zeros((x1.shape[0], 2)).cuda()
        for i in range(x1.shape[0]):
            feature2 = self.pooling_layer1(x1[i, :, :int(x1.shape[2] * bz[i])])
            output[i] = self.fc(feature2.view(feature2.shape[0]))
        return output

    def forward(self, x: Tensor, bz, device) -> Tensor:
        return self._forward_impl(x, bz, device)

class conv2dconv1drpadding(nn.Module):
    def __init__(
        self,
        conv1,
        conv2,
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
            nn.Linear(conv2, 256),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.Dropout(0.5),
            # nn.Linear(64, 32),
            # nn.Dropout(0.2),
            nn.Linear(128,num_classes))
        self.conv1d = nn.Sequential(
            nn.Conv1d(512, conv1, 3, 2, 1, bias=False),
            nn.BatchNorm1d(conv1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv1, conv2, 5, 2, 1, bias=False),
            # nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
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
        pooled_features = torch.squeeze(feature1,dim=2)
        x1 = self.conv1d(pooled_features)
        output = torch.zeros((x1.shape[0], 2)).cuda()
        for i in range(x1.shape[0]):
            sample = x1[i, :, :int(x1.shape[2] * bz[i])].unsqueeze(0)  # 增加batch_size维度
            feature2 = self.pooling_layer1(sample)
            # feature2 = self.pooling_layer1(x1[i, :, :int(x1.shape[2] * bz[i])]) #原来的
            # 使用 .squeeze() 方法移除所有大小为1的维度，这会将形状从 [1, 2048, 1] 变为 [1, 2048]
            feature2_squeezed = feature2.squeeze()
            # 现在 feature2_squeezed 的形状是 [1, 2048]，可以被全连接层接受
            output[i] = self.fc(feature2_squeezed)
            # output[i] = self.fc(feature2.view(feature2.shape[0])) #原来代码
        return output

    def forward(self, x: Tensor, bz, device) -> Tensor:
        return self._forward_impl(x, bz, device)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(3407)
#
# mo=models.resnet18(pretrained=False)
# print(mo._modules.items())

# mo=conv2dconv1dpadding([2,2,2,2])
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



