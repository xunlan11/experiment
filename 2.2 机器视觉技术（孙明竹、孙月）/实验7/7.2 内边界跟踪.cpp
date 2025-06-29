#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <cmath>
using namespace cv;
using namespace std;

const double M_PI = 3.14159265358979323846;

// �ڱ߽���� 
void myFindContours(Mat& image, vector<vector<Point>>& contours, vector<Vec4i>& hierarchy) {
    Mat visited = Mat::zeros(image.size(), CV_8UC1);  // ���ʱ�Ǿ���1Ϊ�ѷ���
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (image.at<uchar>(y, x) == 1 && visited.at<uchar>(y, x) == 0) {
                vector<Point> contour;
                Point currentPoint(x, y);
                int dir = 7; 
                while (true) {
                    visited.at<uchar>(currentPoint.y, currentPoint.x) = 1; 
                    contour.push_back(currentPoint);
                    bool foundNextPoint = false;
                    for (int i = 0; i < 8; i++) {
                        int newDir = (dir + i) % 8;
                        Point newPoint = currentPoint + Point(
                            static_cast<int>(round(cos(newDir * M_PI / 4.0))),
                            static_cast<int>(round(sin(newDir * M_PI / 4.0)))
                        );
                        if (newPoint.x >= 0 && newPoint.x < image.cols &&
                            newPoint.y >= 0 && newPoint.y < image.rows &&
                            image.at<uchar>(newPoint.y, newPoint.x) == 255 &&
                            visited.at<uchar>(newPoint.y, newPoint.x) == 0) {
                            currentPoint = newPoint; 
                            dir = (newDir + 4) % 8; 
                            foundNextPoint = true;
                            break;
                        }
                    }
                    if (!foundNextPoint) {
                        break; 
                    }
                }
                contours.push_back(contour); // ���ҵ���������ӵ������б���  
                hierarchy.push_back(Vec4i(-1, -1, -1, -1)); // �򻯵�ʵ�֣������ز�νṹ��Ϣ  
            }
        }
    }
}

void main()
{
    Mat input = imread("1.jpg");
    Mat gray;
    cvtColor(input, gray, COLOR_BGR2GRAY);
    Mat binary;
    int thresholdValue = 127;
    threshold(gray, binary, thresholdValue, 255, cv::THRESH_BINARY);

    Mat InnerContours = binary.clone();
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    myFindContours(InnerContours, contours, hierarchy); 

    // �����ڲ�����  
    for (size_t i = 0; i < contours.size(); i++) {
        if (hierarchy[i][3] >= 0) { // ����ڲ�����  
            drawContours(InnerContours, contours, static_cast<int>(i), Scalar(0, 0, 255), 2, 8, hierarchy, 0);
        }
    }

    imshow("binary", binary);
    imshow("InnerContours", InnerContours);
    waitKey(0);
}