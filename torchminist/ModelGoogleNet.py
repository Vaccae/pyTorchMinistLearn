import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        ##Branch的池化层,用卷积1X1来处理，1X1的卷积可以直接将Channel层数
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 24, kernel_size=1)
        )
        
        ##Branch1X1层
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1)
        )

        ##Branch5x5层, 5X5保持原图像大小需要padding为2，像3x3的卷积padding为1即可
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=5, padding=2)
        )

        ##Branch3x3层
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, padding=1)
        )

    def forward(self, x):
        ##池化层
        branch_pool = self.branch_pool(x)
        ##branch1X1
        branch1x1 = self.branch1x1(x)
        ##Branch5x5
        branch5x5 = self.branch5x5(x)
        ##Branch3x3
        branch5x5 = self.branch3x3(x)

        ##然后做拼接
        outputs = [branch_pool, branch1x1, branch5x5, branch5x5]
        ##dim=1是为了将channel通道数进行统一， 正常是 B,C,W,H  batchsize,channels,width,height
        ##输出通道数这里计算，branch_pool=24， branch1x1=16， branch5x5=24， branch3x3=24
        ##计算结果就是 24+16+24+24 = 88，在下面Net训练时就知道输入是88通道了
        return torch.cat(outputs, dim=1)


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        ##训练的图像为1X28X28,所以输入通道为1,图像转为10通道后再下采样,再使用用Inception
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Inception(10)
        )

        ##训练的通道由上面的Inception输出，上面计算的输出通道为88，所以这里输入通道就为88
        self.conv2 = nn.Sequential(
            nn.Conv2d(88, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Inception(20)
        )

        ##全链接层,1408是结过上面的网络全部计算出来的，不用自己算，可以输入的时候看Error来修改
        self.fc = nn.Sequential(
            nn.Linear(1408, 10)
        )

        ##定义损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)

        #x = x.view(in_size, -1)
        ##使用x.view导出的onnx在OpenCV中推理会报错，需要改为x.flatten(1)才可以
        x = x.flatten(1)
        x = self.fc(x)
        return x

