'''
Author: Vaccae
Date: 2022-11-13 14:26:24
LastEditors: Vaccae
LastEditTime: 2022-11-15 16:30:38
FilePath: \torchminist\ModelConv2d.py
Description: 

Copyright (c) 2022 by Vaccae 3657447@qq.com, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dNet(nn.Module):
    def __init__(self):
        super(Conv2dNet, self).__init__()
        ##源图像为1 * 28 * 28
        ##从一层channel转为输出5层, 卷积和是3，所以输出的宽和高就是28-3+1 = 26
        ##输出的图像就是 5 * 26 * 26, 然后做pooling下采样2， 变为 5 * 13 * 13
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        ##从上层channel的5转为输出10层, 卷积和是3，所以输出的宽和高就是13-3+1 = 11
        ##输出的图像就是 10 * 11 * 11, 然后做pooling下采样2， 变为 10 * 5 * 5
        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        ##从上层channel的10转为输出20层, 卷积和是3，所以输出的宽和高就是5-3+1 = 3
        ##本层不再做池化了，所以输出的图像就是 20 * 3 * 3,
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3),
            nn.ReLU()
        )

        ##经过上面的图像20 * 3 * 3 = 180,进行全连接，我们多做几层将输出层降到10
        self.fc = nn.Sequential(
            nn.Linear(180, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )     

        ##定义损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

    
    def forward(self, x):
        in_size = x.size(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        #x = x.view(in_size, -1)
        ##使用x.view导出的onnx在OpenCV中推理会报错，需要改为x.flatten(1)才可以
        x = x.flatten(1)
        x = self.fc(x)

        return x