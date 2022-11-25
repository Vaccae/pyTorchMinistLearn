#pragma once
#include<iostream>
#include<chrono>
#include<time.h>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn/dnn.hpp>

using namespace cv;
using namespace std;

//����iType  0-��ȡͼƬ����   1-ʹ��DNN����
int iType = 0;

dnn::Net net;

//�������
void SortRect(vector<Rect>& inputrects) {
	for (int i = 0; i < inputrects.size(); ++i) {
		for (int j = i; j < inputrects.size(); ++j) {
			//˵��˳�����Ϸ������ﲻ�ñ�
			if (inputrects[i].y + inputrects[i].height < inputrects[i].y) {

			}
			//ͬһ��
			else if (inputrects[i].y <= inputrects[j].y + inputrects[j].height) {
				if (inputrects[i].x > inputrects[j].x) {
					swap(inputrects[i], inputrects[j]);
				}
			}
			//��һ��
			else if (inputrects[i].y > inputrects[j].y + inputrects[j].height) {
				swap(inputrects[i], inputrects[j]);
			}
		}
	}
}

//����DNN����MINISTͼ�񣬷�ֹ������ͼ��ֱ��תΪ28*28����
void DealInputMat(Mat& src, int row = 28, int col = 28, int tmppadding = 5) {
	int w = src.cols;
	int h = src.rows;
	//��ͼ��Ŀ�߶Աȣ����д�������padding����ɫ����֤ͼ��ӽ������Σ���������28*28��������ʧ��
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

// ��ȡ��ʱϵͳʱ��
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
	//����onnx�ļ�
	string onnxfile = "D:/Business/DemoTEST/CPP/OpenCVMinistDNN/torchminist/ResNet.onnx";

	//����ͼƬ�ļ�
	string testfile = "D:/Business/DemoTEST/CPP/OpenCVMinistDNN/testpic/train9.png";

	//��ȡ��ͼƬ����λ��
	string savefile = "D:/Business/DemoTEST/CPP/OpenCVMinistDNN/findcontoursMat";

	if (iType == 1) {
		net = dnn::readNetFromONNX(onnxfile);
		if (net.empty()) {
			cout << "����Onnx�ļ�ʧ�ܣ�" << endl;
			return -1;
		}
	}

	//��ȡͼƬ���Ҷȣ���˹ģ��
	Mat src = imread(testfile);
	//����Դͼ
	Mat backsrc;
	src.copyTo(backsrc);
	cvtColor(src, src, COLOR_BGR2GRAY);
	GaussianBlur(src, src, Size(3, 3), 0.5, 0.5);
	//��ֵ��ͼƬ��ע����THRESH_BINARY_INV��Ϊ�ڵװ��֣���ӦMINIST
	threshold(src, src, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	//�����˴�����ֹ��д������û������������������3�����ʹ���
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	//���뿪������ȥ���
	morphologyEx(src, src, MORPH_OPEN, kernel, Point(-1, -1));
	morphologyEx(src, src, MORPH_DILATE, kernel, Point(-1, -1), 3);
	imshow("src", src);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Rect> rects;

	//��������
	findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	for (int i = 0; i < contours.size(); ++i) {
		RotatedRect rect = minAreaRect(contours[i]);
		Rect outrect = rect.boundingRect();
		//���뵽�����б���
		rects.push_back(outrect);
	}

	//�������ң����ϵ�������
	SortRect(rects);
	//Ҫ�����ͼ�����
	for (int i = 0; i < rects.size(); ++i) {
		Mat tmpsrc = src(rects[i]);
		DealInputMat(tmpsrc);

		if (iType == 1) {
			//Mat inputBlob = dnn::blobFromImage(tmpsrc, 0.3081, Size(28, 28), Scalar(0.1307), false, false);
			Mat inputBlob = dnn::blobFromImage(tmpsrc, 1, Size(28, 28), Scalar(), false, false);

			//�������ֵ
			net.setInput(inputBlob, "input");
			//Ԥ���� 
			Mat output = net.forward("output");

			//���ҳ��������������ֵ
			Point maxLoc;
			minMaxLoc(output, NULL, NULL, NULL, &maxLoc);

			cout << "Ԥ��ֵ��" << maxLoc.x << endl;

			//������ȡͼ��λ�ã�����ʾʶ�������
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
