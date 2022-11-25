//
// Created by 36574 on 2022/11/23.
//
#pragma once
#ifndef OPENCVMINIST4ANDROID_DNNUTIL_H
#define OPENCVMINIST4ANDROID_DNNUTIL_H

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class dnnUtil {
private:
    string _onnxdesc;
    dnn::Net _net;
public:
    //初始化Dnn
    bool InitDnnNet(string onnxdesc);
    //推理
    Mat DnnPredict(Mat src);
};


#endif //OPENCVMINIST4ANDROID_DNNUTIL_H
