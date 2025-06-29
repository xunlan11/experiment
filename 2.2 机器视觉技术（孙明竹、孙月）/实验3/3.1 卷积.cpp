#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;

Mat convolution(Mat img, Mat kernel) 
{
	// ��óߴ�
	auto Row = img.rows;
	auto Col = img.cols;
	auto row = kernel.rows;
	auto col = kernel.cols;

	// ת��
	img.convertTo(img, CV_32F);
	normalize(img, img, 0, 1, NORM_MINMAX, CV_32F);

	// ��ʼ�����ͼ��
	Mat Filter2D(Row, Col, CV_32F);
	Filter2D.setTo(Scalar(0));

	// ���
	for (int i = row / 2; i < Row - row / 2; i++) {
		for (int j = col / 2; j < Col - col / 2; j++) {
			for (int m = 0; m < row; m++) {
				for (int n = 0; n < col; n++) {
					Filter2D.at<float>(i, j) += img.at<float>(i - col / 2 + m, j - row / 2 + n) * kernel.at<float>(m, n);
				}
			}
		}
	}

	// ��һ��
	normalize(Filter2D, Filter2D, 0, 255, NORM_MINMAX, CV_8U);
	return Filter2D;
}

Mat myConv(Mat img)
{
	// ���ͼ��
	Mat Conv_img;

    // ��������  
    Mat kernel = (Mat_<float>(3, 3) <<
         1,  1,  1,
		 0,  0,  0,
		-1, -1, -1);
    
    // ���о��  
	Mat Filter2D = convolution(img, kernel);
        
	// ���ؾ�����
	return Conv_img;
}

void main()
{
	Mat input = imread("testimg.jpg");
	
	// ��ɫͼתΪ�Ҷ�ͼ
	Mat gray;
	cvtColor(input, gray, COLOR_BGR2GRAY);

	// ͼ��������
	Mat Conv_img = myConv(gray);

	imshow("Conv_img", Conv_img);
	waitKey(0);
}