# pyTorchMinistLearn综合练习
## 使用pyTorch训练Minist数据集，导出ONNX模型文件，再使用C++ OpenCV DNN进行推理，并用Android做了一个手写板，用于手写数字识别的Demo
<br>

### 由于源码中数据集及生成的训练图片超过100M，所以这里只上传的代码部分，完整的项目包括数据集可以通过百度网度下载：
### 链接：https://pan.baidu.com/s/1wh--rE9m69DoWOAIZWxc3w 
提取码：82d5 

<br>

### 文件目录
```
├─datasets      训练数据集
│  ├─mnist         Minist数据集
│  └─mydata        自己生成的训练数据集图片
├─findcontoursMat     生成训练图片临时存放目录
├─OpenCVMinist4Android  Android手写数字识别的Demo，用的kotlin+ndk OpenCV
│  │  ├─libs            OpenCV 4.6的动态库
│  │  │  ├─arm64-v8a
│  │  │  ├─armeabi-v7a
│  │  │  └─x86
│  │  └─src             Android的Demo源码
├─OpenCVMinistDNN       VS2022写的C++  OpenCV识别集生成训练图片的Demo
├─testpic               测试图片
└─torchminist           pyTorch的训练源码
```
<br>

## Demo对应的学习文章，关注公众号《微卡智享》获取更多信息

### [pyTorch入门（一）——Minist手写数据识别训练全连接网络](https://mp.weixin.qq.com/s/zo_BadXJqcTn0PsAhYii0A)  
### [pyTorch入门（二）——常用网络层函数及卷积神经网络训练](https://mp.weixin.qq.com/s/rTDuNh2Y2K0l5F4En6rV0g)  
### [pyTorch入门（三）——GoogleNet和ResNet训练](https://mp.weixin.qq.com/s/Ak7gbREwawgM4Y4-l6xQww)  
### [pyTorch入门（四）——导出Minist模型，C++ OpenCV DNN进行识别](https://mp.weixin.qq.com/s/BDV0SsyvZ6mKrKnCmX3igA)  
### [pyTorch入门（五）——训练自己的数据集](https://mp.weixin.qq.com/s/GZVfaxhWl5s5jUCYppO9UA)  
