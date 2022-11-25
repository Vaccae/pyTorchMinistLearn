#pragma once
#include<iostream>
#include<chrono>
#include<time.h>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn/dnn.hpp>

using namespace cv;
using namespace std;

//参数iType  0-提取图片保存   1-使用DNN推理
int iType = 0;

dnn::Net net;

//排序矩形
void SortRect(vector<Rect>& inputrects) {
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
void DealInputMat(Mat& src, int row = 28, int col = 28, int tmppadding = 5) {
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

// 获取当时系统时间
const string GetCurrentSystemTime()
{
	auto t = chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	struct tm ptm { 60, 59, 23, 31, 11, 1900, 6, 365, -1 };
	_localtime64_s(&ptm, &t);
	char date[60] = { 0 };
	sprintf_s(date, "%d%02d%02d%02d%02d%02d",
		(int)ptm.tm_year + 1900, (int)ptm.tm_mon + 1, (int)ptm.tm_mday,
		(int)ptm.tm_hour, (int)ptm.tm_min, (int)ptm.tm_sec);
	return move(std::string(date));
}

int main(int argc, char** argv) {
	//定义onnx文件
	string onnxfile = "D:/Business/DemoTEST/CPP/OpenCVMinistDNN/torchminist/ResNet.onnx";

	//测试图片文件
	string testfile = "D:/Business/DemoTEST/CPP/OpenCVMinistDNN/testpic/train9.png";

	//提取的图片保存位置
	string savefile = "D:/Business/DemoTEST/CPP/OpenCVMinistDNN/findcontoursMat";

	if (iType == 1) {
		net = dnn::readNetFromONNX(onnxfile);
		if (net.empty()) {
			cout << "加载Onnx文件失败！" << endl;
			return -1;
		}
	}

	//读取图片，灰度，高斯模糊
	Mat src = imread(testfile);
	//备份源图
	Mat backsrc;
	src.copyTo(backsrc);
	cvtColor(src, src, COLOR_BGR2GRAY);
	GaussianBlur(src, src, Size(3, 3), 0.5, 0.5);
	//二值化图片，注意用THRESH_BINARY_INV改为黑底白字，对应MINIST
	threshold(src, src, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	//做彭账处理，防止手写的数字没有连起来，这里做了3次膨胀处理
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	//加入开运算先去燥点
	morphologyEx(src, src, MORPH_OPEN, kernel, Point(-1, -1));
	morphologyEx(src, src, MORPH_DILATE, kernel, Point(-1, -1), 3);
	imshow("src", src);

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
	SortRect(rects);
	//要输出的图像参数
	for (int i = 0; i < rects.size(); ++i) {
		Mat tmpsrc = src(rects[i]);
		DealInputMat(tmpsrc);

		if (iType == 1) {
			//Mat inputBlob = dnn::blobFromImage(tmpsrc, 0.3081, Size(28, 28), Scalar(0.1307), false, false);
			Mat inputBlob = dnn::blobFromImage(tmpsrc, 1, Size(28, 28), Scalar(), false, false);

			//输入参数值
			net.setInput(inputBlob, "input");
			//预测结果 
			Mat output = net.forward("output");

			//查找出结果中推理的最大值
			Point maxLoc;
			minMaxLoc(output, NULL, NULL, NULL, &maxLoc);

			cout << "预测值：" << maxLoc.x << endl;

			//画出截取图像位置，并显示识别的数字
			rectangle(backsrc, rects[i], Scalar(255, 0, 255));
			putText(backsrc, to_string(maxLoc.x), Point(rects[i].x, rects[i].y), FONT_HERSHEY_PLAIN, 5, Scalar(255, 0, 255), 1, -1);
		}
		else {
			string filename = savefile + "/" + GetCurrentSystemTime() + "-" + to_string(i) + ".jpg";
			cout << filename << endl;
			imwrite(filename, tmpsrc);
		}
	}

	imshow("backsrc", backsrc);


	waitKey(0);
	return 0;
}
