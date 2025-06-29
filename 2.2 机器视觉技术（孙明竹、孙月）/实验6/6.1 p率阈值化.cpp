#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;

Mat myThresholdP(Mat img)
{
	Mat Hist;

	// 计算直方图
	int histSize = 256; 
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true, accumulate = false;
	calcHist(&img, 1, 0, Mat(), Hist, 1, &histSize, &histRange, uniform, accumulate);

	float P = 0.5; //设定P值
	float T = 0; //经由直方图得到的阈值
	float sum_pixels = 0;
	for (int i = 0; i < histSize; i++) {
		sum_pixels += Hist.at<float>(i);
		if (sum_pixels >= img.rows * img.cols * P) {
			break;
		}
		T = i;
	}

	// 根据T对图像进行二值化处理
	Mat ThresholdPImg = img.clone();
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (ThresholdPImg.at<uchar>(i, j) >= T) {
				ThresholdPImg.at<uchar>(i, j) = 255;
			}
			else {
				ThresholdPImg.at<uchar>(i, j) = 0;
			}
		}
	}

	//返回p率阈值化结果
	return ThresholdPImg;
}

void main()
{
	Mat input = imread("testimg.jpg");
	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	//灰度图p率阈值化
	Mat ThresholdPImg = myThresholdP(gray);

	imshow("ThresholdPImg", ThresholdPImg);
	waitKey(0);
}