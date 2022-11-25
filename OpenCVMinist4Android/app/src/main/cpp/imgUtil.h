//
// Created by 36574 on 2022/11/25.
//
#pragma once
#ifndef OPENCVMINIST4ANDROID_IMGUTIL_H
#define OPENCVMINIST4ANDROID_IMGUTIL_H

#include <jni.h>
#include <string>
#include <android/log.h>
#include <android/bitmap.h>
#include "opencv2/opencv.hpp"

#define LOG_TAG "imgutil.out"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)


using namespace cv;
using namespace std;

class imgUtil {
public:
    //Bitmap转为Mat
    Mat bitmap2Mat(JNIEnv *env, jobject bmp);
    //获取图像参数
    jobject getBitmapConfig(JNIEnv *env, jobject bmp);
    //Mat转为Bitmap
    jobject mat2Bitmap(JNIEnv *env, cv::Mat &src, bool needPremultiplyAlpha,  jobject bitmap_config);

    //排序矩形
    void sortRect(vector<Rect>& inputrects);
    //处理DNN检测的MINIST图像，防止长方形图像直接转为28*28扁了
    void dealInputMat(Mat& src, int row = 28, int col = 28, int tmppadding = 5);
};


#endif //OPENCVMINIST4ANDROID_IMGUTIL_H
