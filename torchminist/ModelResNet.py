'''
Author: Vaccae
Date: 2022-11-11 15:15:16
LastEditors: Vaccae
LastEditTime: 2022-11-15 16:17:35
FilePath: \torchminist\ModelResNet.py
Description: 

Copyright (c) 2022 by Vaccae 3657447@qq.com, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBolck(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBolck, self).__init__()

        self.channels = in_channels
        ##确保输入层和输出层一样图像大小，所以padding=1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        ##第二层只有一个卷积，所以不用nn.Sequential了
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        ##求出第一层
        y = self.conv1(x)
        ##求出第二层
        y = self.conv2(y)
        ##通过加上原来X后再用激活，防止梯度归零
        y = F.relu(x+y)
        return y


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        ##第一层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualBolck(16)
        )
        ##第二层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualBolck(32)
        )
        ##全连接层
        self.fc = nn.Linear(512, 10)
        ##定义损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        #in_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)

        #x = x.view(in_size, -1)
        ##使用x.view导出的onnx在OpenCV中推理会报错，需要改为x.flatten(1)才可以
        x = x.flatten(1)
        x = self.fc(x)
        return x