import torch
import torch.nn.functional as F

##Minist的图像为1X28X28的
class LinearNet(torch.nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        ##784是图像为 1X28X28，通道X宽度X高度，然后总的输入就是784
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
        ##定义损失函数
        self.criterion = torch.nn.CrossEntropyLoss()


    def forward(self, x):
        ##将输入的图像矩阵改为N行784列
        #x = x.view(-1, 784)
        ##使用x.view导出的onnx在OpenCV中推理会报错，需要改为x.flatten(1)才可以
        x = x.flatten(1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        ##最后一层激活在损失函数中加入了，这里直接输出，不要加上rule了
        return self.l5(x)