#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// 색 검출 -> 원 검출
int main(int argc, char** argv) {

    //VideoCapture capture("./Traffic_Light.mp4");
    VideoCapture capture("./Traffic_Light2.mov");

    if (!capture.isOpened()) {
        printf("Can't open the video");
        return 0;
    }

    //opencv 이미지 변수
    Mat frame;

    while (1) {
        capture >> frame;
        if (frame.empty()) break;

        //frame = frame(Range(0,frame.size().height * 2/3), Range::all());
        resize(frame, frame, Size(600, 320));

        Mat hsv, binary_red1, binary_red2, binary_green, binary_blue;

        // HSV 변환
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // 지정한 HSV 범위를 이용해서 mask 영상
        //inRange(hsv, Scalar(0, 150, 0), Scalar(10, 255, 255), binary_red1);
        //inRange(hsv, Scalar(160, 150, 0), Scalar(180, 255, 255), binary_red2);
        //inRange(hsv, Scalar(40, 150, 0), Scalar(100, 255, 255), binary_green);
        inRange(hsv, Scalar(0, 150, 0), Scalar(10, 255, 255), binary_red1);
        inRange(hsv, Scalar(170, 150, 0), Scalar(180, 255, 255), binary_red2);
        inRange(hsv, Scalar(40, 150, 0), Scalar(100, 255, 255), binary_green);

        // hue 평균값
        float green_hue_mean = mean(hsv, binary_green)[0];
        float red1_hue_mean = mean(hsv, binary_red1)[0];
        float red2_hue_mean = mean(hsv, binary_red2)[0];

        printf("red1 = %.2f\n", mean(hsv, binary_red1)[0]);
        printf("red2 = %.2f\n", mean(hsv, binary_red2)[0]);
        printf("green = %.2f\n", mean(hsv, binary_green)[0]);

        Mat binary;

        if (green_hue_mean > 0) {
            //printf("green 검출\n");
            binary_green.copyTo(binary);
        }
        else if (red1_hue_mean > 0) {
            //printf("red 검출 \n");
            binary_red1.copyTo(binary);
        }
        else if (red2_hue_mean > 0) {
            //printf("red 검출 \n");
            binary_red2.copyTo(binary);
        }

        // 노이즈 제거
        erode(binary, binary, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        // 원검출을 위해 확장
        dilate(binary, binary, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)), Point(-1, -1), 10);

        Mat color_range;
        add(frame, frame, color_range, binary);

        imshow("color_range", color_range);

        Mat gray;
        cvtColor(color_range, gray, CV_BGR2GRAY);
        Mat blur;
        GaussianBlur(gray, blur, Size(0, 0), 1.0);

        //imshow("blur", blur);

        // 원 검출
        vector<Vec3f> circles;
        HoughCircles(blur, circles, CV_HOUGH_GRADIENT, 1, 50, 120, 50, 10, 90);
        //HoughCircles(blur, circles, CV_HOUGH_GRADIENT, 1, 50, 120, 60, 40, 100);

        Mat dst;
        frame.copyTo(dst);

        for (size_t i = 0; i < circles.size(); i++) {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            circle(frame, center, radius, Scalar(0, 0, 255), 3, 8, 0);

            string color = "none";
            string hue_mean = "";
            if (green_hue_mean > 40 && green_hue_mean < 100) {
            //if (green_hue_mean > 40 && green_hue_mean < 80) {
                color = "green";
                hue_mean = to_string(green_hue_mean);
            }
            else if (red1_hue_mean > 0 && red1_hue_mean < 10) {
                color = "red";
                hue_mean = to_string(red1_hue_mean);
            }
            else if (red2_hue_mean > 170) {
                color = "red";
                hue_mean = to_string(red2_hue_mean);
            }
            putText(frame, color, center, CV_FONT_HERSHEY_SIMPLEX, 0.75, Scalar::all(0));

            Point center_10(cvRound(circles[i][0]), cvRound(circles[i][1]) + 20);
            putText(frame, hue_mean, center_10, CV_FONT_HERSHEY_SIMPLEX, 0.75, Scalar::all(0));

            imshow("frame", frame);
        }
        /**/
        if (waitKey(1) == 27) break;
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}