'''
Author: Vaccae
Date: 2022-11-15 09:46:47
LastEditors: Vaccae
LastEditTime: 2022-11-21 11:37:50
FilePath: \torchminist\trainModel.py
Description: 

Copyright (c) 2022 by Vaccae 3657447@qq.com, All Rights Reserved. 
'''
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from ModelLinearNet import LinearNet
from ModelConv2d import Conv2dNet
from ModelGoogleNet import GoogleNet
from ModelResNet import ResNet


batch_size = 64
##设置本次要训练用的模型
train_name = 'ResNet'
print("train_name:" + train_name)
##设置模型保存名称
savemodel_name = train_name + ".pt"
print("savemodel_name:" + savemodel_name)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
]) ##Normalize 里面两个值0.1307是均值mean， 0.3081是标准差std，计算好的直接用了

##训练数据集位置，如果不存在直接下载
train_dataset = datasets.MNIST(
    root = '../datasets/mnist', 
    train = True,
    download = True,
    transform = transform
)
##读取训练数据集
train_dataloader = DataLoader(
    dataset= train_dataset,
    shuffle=True,
    batch_size=batch_size
)
##测试数据集位置，如果不存在直接下载
test_dataset = datasets.MNIST(
    root= '../datasets/mnist',
    train= False,
    download=True,
    transform= transform
)
##读取测试数据集
test_dataloader = DataLoader(
    dataset= test_dataset,
    shuffle= True,
    batch_size=batch_size
)


##设置选择训练模型,因为python用的是3.9,用不了match case语法
def switch(train_name):
    if train_name == 'LinearNet':
        return LinearNet()
    elif train_name == 'Conv2dNet':
        return Conv2dNet()
    elif train_name == 'GoogleNet':
        return GoogleNet()
    elif train_name == 'ResNet':
        return ResNet()


##定义训练模型
class Net(torch.nn.Module):
    def __init__(self, train_name):
        super(Net, self).__init__()
        self.model = switch(train_name= train_name)
        self.criterion = self.model.criterion

    def forward(self, x):
        x = self.model(x)
        return x