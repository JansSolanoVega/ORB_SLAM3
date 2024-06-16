#include <iostream>
#include<opencv2/opencv.hpp>

#include<math.h>
#include "yolov8.h"
#include "yolov8_seg.h"
#include<time.h>
#define  VIDEO_OPENCV //if define, use opencv for video.

using namespace std;
using namespace cv;
using namespace dnn;

template<typename _Tp>
int yolov8(_Tp& task, cv::Mat& img, std::string& model_path)
{


	cv::dnn::Net net;
	if (task.ReadModel(net, model_path, true)) {
		std::cout << "read net ok!" << std::endl;
	}
	else {
		return -1;
	}
	//生成随机颜色
	std::vector<cv::Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(cv::Scalar(b, g, r));
	}
	std::vector<OutputParams> result;

	if (task.Detect(img, net, result)) {
		DrawPred(img, result, task._className, color);
		
	}
	else {
		std::cout << "Detect Failed!" << std::endl;
	}
	system("pause");
	return 0;
}

template<typename _Tp>
int video_demo(_Tp& task, std::string& model_path)
{
	std::vector<cv::Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(cv::Scalar(b, g, r));
	}
	std::vector<OutputParams> result;
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cout << "open capture failured!" << std::endl;
		return -1;
	}
	cv::Mat frame;
	bool isPose = false;
	PoseParams poseParams;
#ifdef VIDEO_OPENCV
	cv::dnn::Net net;
	if (task.ReadModel(net, model_path, true)) {
		std::cout << "read net ok!" << std::endl;
	}
	else {
		std::cout << "read net failured!" << std::endl;
		return -1;
	}
#endif

	while (true)
	{

		cap.read(frame);
		if (frame.empty())
		{
			std::cout << "read to end" << std::endl;
			break;
		}
		result.clear();
#ifdef VIDEO_OPENCV

		if (task.Detect(frame, net, result)) {

			if (isPose)
				DrawPredPose(frame, result, poseParams,true);
			else
				DrawPred(frame, result, task._className, color,true);
	
		}
#endif
		int k = waitKey(10);
		if (k == 27) { //esc 
			break;
		}

	}
	cap.release();

	system("pause");

	return 0;
}


int main() {

	std::string img_path = "./images/zidane.jpg";
	std::string model_path_seg = "dynamic/models/yolov8s-seg.onnx";

	cv::Mat src = imread(img_path);
	cv::Mat img = src.clone();

	Yolov8				task_detect_ocv;
	Yolov8Seg			task_segment_ocv;


	// img = src.clone();
	// yolov8(task_segment_ocv,img,model_path_seg);   //yolov8 opencv segment

#ifdef VIDEO_OPENCV
	video_demo(task_segment_ocv, model_path_seg);
#endif
	return 0;
}


