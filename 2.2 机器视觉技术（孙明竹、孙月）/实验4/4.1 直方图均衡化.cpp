#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;

Mat myEqualizeHist(Mat img)
{
	Mat Hist;
	
	// ����ֱ��ͼ
	int histSize = 256; 
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true, accumulate = false;
	calcHist(&img, 1, 0, Mat(), Hist, 1, &histSize, &histRange, uniform, accumulate);

	// �����ۻ�ֱ��ͼ
	Mat HC;
	Hist.copyTo(HC);
	for (int i = 1; i < histSize; i++){
		HC.at<float>(i) += HC.at<float>(i - 1);
	}

	// ֱ��ͼ���⻯
	Mat equalizedHist = Mat::zeros(1, histSize, CV_32FC1);
	for (int i = 0; i < histSize; i++){
		equalizedHist.at<float>(i) = round(HC.at<float>(i) * (histSize - 1) / img.total());
	}

	return equalizedHis;
}

void main()
{
	Mat input = imread("testimg.jpg");
	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	//ֱ��ͼ���⻯
	Mat EqualizedImg = myEqualizeHist(gray);

	imshow("input", input);
	imshow("EqualizedImg", EqualizedImg);
	waitKey(0);
}