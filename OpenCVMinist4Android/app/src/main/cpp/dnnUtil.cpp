//
// Created by 36574 on 2022/11/23.
//

#include "dnnUtil.h"

bool dnnUtil::InitDnnNet(string onnxdesc) {
    _onnxdesc = onnxdesc;

    _net = dnn::readNetFromONNX(_onnxdesc);
    _net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    return !_net.empty();
}

Mat dnnUtil::DnnPredict(Mat src) {
    Mat inputBlob = dnn::blobFromImage(src, 1, Size(28, 28), Scalar(), false, false);

    //输入参数值
    _net.setInput(inputBlob, "input");
    //预测结果
    Mat output = _net.forward("output");

    return output;
}
