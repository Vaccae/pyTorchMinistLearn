//
// Created by 36574 on 2022/11/25.
//

#include "imgUtil.h"

//Bitmap转为Mat
Mat imgUtil::bitmap2Mat(JNIEnv *env, jobject bmp) {

    Mat src;
    AndroidBitmapInfo bitmapInfo;
    void *pixelscolor;
    int ret;
    try {
        //获取图像信息，如果返回值小于0就是执行失败
        if ((ret = AndroidBitmap_getInfo(env, bmp, &bitmapInfo)) < 0) {
            LOGI("AndroidBitmap_getInfo failed! error-%d", ret);
            return src;
        }

        //判断图像类型是不是RGBA_8888类型
        if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOGI("BitmapInfoFormat error");
            return src;
        }

        //获取图像像素值
        if ((ret = AndroidBitmap_lockPixels(env, bmp, &pixelscolor)) < 0) {
            LOGI("AndroidBitmap_lockPixels() failed ! error=%d", ret);
            return src;
        }

        //生成源图像
        src = Mat(bitmapInfo.height, bitmapInfo.width, CV_8UC4, pixelscolor);

        return src;
    } catch (Exception e) {
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return src;
    } catch (...) {
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {bitmap2Mat}");
        return src;
    }
}

//获取Bitmap的参数
jobject imgUtil::getBitmapConfig(JNIEnv *env, jobject bmp) {
    //获取原图片的参数
    jclass java_bitmap_class = (jclass) env->FindClass("android/graphics/Bitmap");
    jmethodID mid = env->GetMethodID(java_bitmap_class, "getConfig",
                                     "()Landroid/graphics/Bitmap$Config;");
    jobject bitmap_config = env->CallObjectMethod(bmp, mid);
    return bitmap_config;
}

//Mat转为Bitmap
jobject
imgUtil::mat2Bitmap(JNIEnv *env, Mat &src, bool needPremultiplyAlpha, jobject bitmap_config) {

    jclass java_bitmap_class = (jclass) env->FindClass("android/graphics/Bitmap");
    jmethodID mid = env->GetStaticMethodID(java_bitmap_class, "createBitmap",
                                           "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    jobject bitmap = env->CallStaticObjectMethod(java_bitmap_class,
                                                 mid, src.size().width, src.size().height,
                                                 bitmap_config);
    AndroidBitmapInfo info;
    void *pixels = 0;

    try {
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);

        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (src.type() == CV_8UC1) {
                cvtColor(src, tmp, cv::COLOR_GRAY2RGBA);
            } else if (src.type() == CV_8UC3) {
                cvtColor(src, tmp, cv::COLOR_RGB2BGRA);
            } else if (src.type() == CV_8UC4) {
                if (needPremultiplyAlpha) {
                    cvtColor(src, tmp, cv::COLOR_RGBA2mRGBA);
                } else {
                    src.copyTo(tmp);
                }
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if (src.type() == CV_8UC1) {
                cvtColor(src, tmp, cv::COLOR_GRAY2BGR565);
            } else if (src.type() == CV_8UC3) {
                cvtColor(src, tmp, cv::COLOR_RGB2BGR565);
            } else if (src.type() == CV_8UC4) {
                cvtColor(src, tmp, cv::COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return bitmap;
    } catch (Exception e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return bitmap;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return bitmap;
    }
}

//排序矩形
void imgUtil::sortRect(vector<Rect> &inputrects) {
    for (int i = 0; i < inputrects.size(); ++i) {
        for (int j = i; j < inputrects.size(); ++j) {
            //说明顺序在上方，这里不用变
            if (inputrects[i].y + inputrects[i].height < inputrects[i].y) {

            }
                //同一排
            else if (inputrects[i].y <= inputrects[j].y + inputrects[j].height) {
                if (inputrects[i].x > inputrects[j].x) {
                    swap(inputrects[i], inputrects[j]);
                }
            }
                //下一排
            else if (inputrects[i].y > inputrects[j].y + inputrects[j].height) {
                swap(inputrects[i], inputrects[j]);
            }
        }
    }
}

//处理DNN检测的MINIST图像，防止长方形图像直接转为28*28扁了
void imgUtil::dealInputMat(Mat &src, int row, int col, int tmppadding) {
    int w = src.cols;
    int h = src.rows;
    //看图像的宽高对比，进行处理，先用padding填充黑色，保证图像接近正方形，这样缩放28*28比例不会失衡
    if (w > h) {
        int tmptopbottompadding = (w - h) / 2 + tmppadding;
        copyMakeBorder(src, src, tmptopbottompadding, tmptopbottompadding, tmppadding, tmppadding,
                       BORDER_CONSTANT, Scalar(0));
    }
    else {
        int tmpleftrightpadding = (h - w) / 2 + tmppadding;
        copyMakeBorder(src, src, tmppadding, tmppadding, tmpleftrightpadding, tmpleftrightpadding,
                       BORDER_CONSTANT, Scalar(0));

    }
    resize(src, src, Size(row, col));
}
