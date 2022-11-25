'''
Author: Vaccae
Date: 2022-11-10 09:19:56
LastEditors: Vaccae
LastEditTime: 2022-11-15 16:19:32
FilePath: \torchminist\train.py
Description: 

Copyright (c) 2022 by Vaccae 3657447@qq.com, All Rights Reserved. 
'''
import torch
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from pylab import mpl
import trainModel as tm

##训练轮数
epoch_times = 10

##设置初始预测率,用于判断高于当前预测率的保存模型
toppredicted = 0.0

##设置学习率
learnrate = 0.01 
##设置动量值，如果上一次的momentnum与本次梯度方向是相同的，梯度下降幅度会拉大，起到加速迭代的作用
momentnum = 0.5

##生成图用的数组
##预测值
predict_list = []
##训练轮次值
epoch_list = []
##loss值
loss_list = []

model = tm.Net(tm.train_name)
##加入判断是CPU训练还是GPU训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

##优化器 
optimizer = optim.SGD(model.parameters(), lr= learnrate, momentum= momentnum)
# optimizer = optim.NAdam(model.parameters(), lr= learnrate)

##训练函数
def train(epoch):
    running_loss = 0.0
    current_train = 0.0
    model.train()
    for batch_idx, data in enumerate(tm.train_dataloader, 0):
        inputs, target = data
        ##加入CPU和GPU选择
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()

        #前馈，反向传播，更新
        outputs = model(inputs)
        loss = model.criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        ##计算每300次打印一次学习效果
        if batch_idx % 300 == 299:
            current_train = current_train + 0.3
            current_epoch = epoch + 1 + current_train
            epoch_list.append(current_epoch)
            current_loss = running_loss / 300
            loss_list.append(current_loss)

            print('[%d, %5d] loss: %.3f' % (current_epoch, batch_idx + 1, current_loss))
            running_loss = 0.0


def test():
    correct = 0 
    total = 0
    model.eval()
    ##with这里标记是不再计算梯度
    with torch.no_grad():
        for data in tm.test_dataloader:
            inputs, labels = data
            ##加入CPU和GPU选择
            inputs, labels = inputs.to(device), labels.to(device)


            outputs = model(inputs)
            ##预测返回的是两列，第一列是下标就是0-9的值，第二列为预测值，下面的dim=1就是找维度1（第二列）最大值输出
            _, predicted = torch.max(outputs.data, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    currentpredicted = (100 * correct / total)
    ##用global声明toppredicted,用于在函数内部修改在函数外部声明的全局变量，否则报错
    global toppredicted
    ##当预测率大于原来的保存模型
    if currentpredicted > toppredicted:
        toppredicted = currentpredicted
        torch.save(model.state_dict(), tm.savemodel_name)
        print(tm.savemodel_name+" saved, currentpredicted:%d %%" % currentpredicted)

    predict_list.append(currentpredicted)    
    print('Accuracy on test set: %d %%' % currentpredicted)        

##开始训练
timestart = time.time()
for epoch in range(epoch_times):
    train(epoch)
    test()
timeend = time.time() - timestart
print("use time: {:.0f}m {:.0f}s".format(timeend // 60, timeend % 60))



##设置画布显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
##设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

##创建画布
fig, (axloss, axpredict) = plt.subplots(nrows=1, ncols=2, figsize=(8,6))

#loss画布
axloss.plot(epoch_list, loss_list, label = 'loss', color='r')
##设置刻度
axloss.set_xticks(range(epoch_times)[::1])
axloss.set_xticklabels(range(epoch_times)[::1])

axloss.set_xlabel('训练轮数')
axloss.set_ylabel('数值')
axloss.set_title(tm.train_name+' 损失值')
#添加图例
axloss.legend(loc = 0)

#predict画布
axpredict.plot(range(epoch_times), predict_list, label = 'predict', color='g')
##设置刻度
axpredict.set_xticks(range(epoch_times)[::1])
axpredict.set_xticklabels(range(epoch_times)[::1])
# axpredict.set_yticks(range(100)[::5])
# axpredict.set_yticklabels(range(100)[::5])

axpredict.set_xlabel('训练轮数')
axpredict.set_ylabel('预测值')
axpredict.set_title(tm.train_name+' 预测值')
#添加图例
axpredict.legend(loc = 0)

#显示图像
plt.show()

