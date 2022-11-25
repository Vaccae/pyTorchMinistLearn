#pragma once

#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include "dnnUtil.h"
#include "imgUtil.h"

#define LOG_TAG "System.out"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using namespace cv;
using namespace std;

dnnUtil _dnnUtil;
imgUtil _imgUtil = imgUtil();

extern "C"
JNIEXPORT jboolean JNICALL
Java_dem_vaccae_opencvminist4android_OpenCVJNI_initOpenCVDNN(JNIEnv *env, jobject thiz,
                                                             jstring onnxfilepath) {
    try {
        string onnxfile = env->GetStringUTFChars(onnxfilepath, 0);
        //初始化DNN
        _dnnUtil = dnnUtil();
        jboolean res = _dnnUtil.InitDnnNet(onnxfile);

        return res;
    } catch (Exception e) {
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
    } catch (...) {
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {initOpenCVDNN}");
    }
}
extern "C"
JNIEXPORT jobject JNICALL
Java_dem_vaccae_opencvminist4android_OpenCVJNI_ministDetector(JNIEnv *env, jobject thiz,
                                                              jobject bmp) {
    try {
        jobject bitmapcofig = _imgUtil.getBitmapConfig(env, bmp);

        string resstr="";

        Mat src = _imgUtil.bitmap2Mat(env, bmp);
        //备份源图
        Mat backsrc;
        src.copyTo(backsrc);
        //将备份的图片从BGRA转为RGB，防止颜色不对
        cvtColor(backsrc, backsrc, COLOR_BGRA2RGB);

        cvtColor(src, src, COLOR_BGRA2GRAY);
        GaussianBlur(src, src, Size(3, 3), 0.5, 0.5);
        //二值化图片，注意用THRESH_BINARY_INV改为黑底白字，对应MINIST
        threshold(src, src, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

        //做彭账处理，防止手写的数字没有连起来，这里做了3次膨胀处理
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        //加入开运算先去燥点
        morphologyEx(src, src, MORPH_OPEN, kernel, Point(-1, -1));
        morphologyEx(src, src, MORPH_DILATE, kernel, Point(-1, -1), 3);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        vector<Rect> rects;

        //查找轮廓
        findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        for (int i = 0; i < contours.size(); ++i) {
            RotatedRect rect = minAreaRect(contours[i]);
            Rect outrect = rect.boundingRect();
            //插入到矩形列表中
            rects.push_back(outrect);
        }

        //按从左到右，从上到下排序
        _imgUtil.sortRect(rects);
        //要输出的图像参数
        for (int i = 0; i < rects.size(); ++i) {
            Mat tmpsrc = src(rects[i]);
            _imgUtil.dealInputMat(tmpsrc);
            //预测结果
            Mat output = _dnnUtil.DnnPredict(tmpsrc);

            //查找出结果中推理的最大值
            Point maxLoc;
            minMaxLoc(output, NULL, NULL, NULL, &maxLoc);

            //返回字符串值
            resstr += to_string(maxLoc.x);

            //画出截取图像位置，并显示识别的数字
            rectangle(backsrc, rects[i], Scalar(0, 0, 255),5);
            putText(backsrc, to_string(maxLoc.x), Point(rects[i].x, rects[i].y), FONT_HERSHEY_PLAIN,
                    5, Scalar(0, 0, 255), 5, -1);

        }

        jobject resbmp = _imgUtil.mat2Bitmap(env, backsrc, false, bitmapcofig);

        //获取MinistResult返回类
        jclass ministresultcls = env->FindClass("dem/vaccae/opencvminist4android/MinistResult");
        //定义MinistResult返回类属性
        jfieldID ministmsg = env->GetFieldID(ministresultcls, "msg", "Ljava/lang/String;");
        jfieldID ministbmp = env->GetFieldID(ministresultcls, "bmp", "Landroid/graphics/Bitmap;");

        //创建返回类
        jobject ministresultobj = env->AllocObject(ministresultcls);
        //设置返回消息
        env->SetObjectField(ministresultobj, ministmsg, env->NewStringUTF(resstr.c_str()));
        //设置返回的图片信息
        env->SetObjectField(ministresultobj, ministbmp, resbmp);


        AndroidBitmap_unlockPixels(env, bmp);

        return ministresultobj;
    } catch (Exception e) {
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
    } catch (...) {
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {bitmap2Mat}");
    }
}



extern "C"
JNIEXPORT jobject JNICALL
Java_dem_vaccae_opencvminist4android_OpenCVJNI_thresholdBitmap(JNIEnv *env, jobject thiz,
                                                               jobject bmp) {
    try {
        jobject bitmapcofig = _imgUtil.getBitmapConfig(env, bmp);

        Mat src = _imgUtil.bitmap2Mat(env, bmp);
        cvtColor(src, src, COLOR_BGRA2GRAY);
        threshold(src, src, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

        jobject resbmp = _imgUtil.mat2Bitmap(env, src, false, bitmapcofig);

        AndroidBitmap_unlockPixels(env, bmp);

        return resbmp;
    } catch (Exception e) {
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
    } catch (...) {
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {bitmap2Mat}");
    }
}