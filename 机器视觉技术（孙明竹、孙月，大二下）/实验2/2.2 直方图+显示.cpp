#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;

Mat myHist(Mat img)
{ 
    // 初始化直方图 
    int bins = 256; 
    Mat histogram(1, bins, CV_32SC1, Scalar(0)); 

    // 计算直方图 
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            int pixelValue = img.at<uchar>(y, x);
            histogram.at<int>(0, pixelValue)++;
        }
    }

    return histogram;
}

void main()
{
	Mat input = imread("testimg.jpg");
	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	Mat Hist = myHist(gray);

    // 创建窗口和图像
    int histW = 512, histH = 400;
    int binW = cvRound(static_cast<double>(histW) / Hist.cols);
    Mat histImage(histH, histW, CV_8UC3, Scalar(255, 255, 255));

    // 归一化  
    double maxVal;
    minMaxLoc(Hist, 0, &maxVal);

    // 绘制直方图  
    for (int i = 0; i < Hist.cols; i++) {
        float binVal = static_cast<float>(Hist.at<int>(0, i));
        int intensity = static_cast<int>((binVal / maxVal) * histH);
        line(histImage, cv::Point(i * binW, histH), Point(i * binW, histH - intensity), Scalar(255, 0, 0), 2, 8, 0);
    }

    // 显示直方图  
    namedWindow("Histogram", WINDOW_AUTOSIZE);
    imshow("Histogram", histImage);

	waitKey(0);
}