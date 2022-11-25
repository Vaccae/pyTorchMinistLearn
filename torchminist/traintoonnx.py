'''
Author: Vaccae
Date: 2022-11-15 08:37:39
LastEditors: Vaccae
LastEditTime: 2022-11-21 12:45:18
FilePath: \torchminist\traintoonnx.py
Description: 

Copyright (c) 2022 by Vaccae 3657447@qq.com, All Rights Reserved. 
'''
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import trainModel as tm

##获取输入参数
data = iter(tm.test_dataloader)
dummy_inputs, labels = next(data)
print(dummy_inputs.shape)

##加载模型
model = tm.Net(tm.train_name)
model.load_state_dict(torch.load(tm.savemodel_name))
print(model)

##加载的模型测试效果
outputs = model(dummy_inputs)
print(outputs)
##预测返回的是两列，第一列是下标就是0-9的值，第二列为预测值，下面的dim=1就是找维度1（第二列）最大值输出
_, predicted = torch.max(outputs.data, dim=1)
print(_)
print(predicted)
outlabels = predicted.numpy().tolist()
print(outlabels)

##定义输出输出的参数名
input_name = ["input"]
output_name = ["output"]

onnx_name = tm.train_name + '.onnx'

torch.onnx.export(
    model,
    dummy_inputs,
    onnx_name,
    verbose=True,
    input_names=input_name,
    output_names=output_name
)

