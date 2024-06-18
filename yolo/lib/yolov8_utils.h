#pragma once
#include<iostream>
#include <numeric>
#include<opencv2/opencv.hpp>
//old line: #include<io.h>
#ifdef _WIN32
    #include <io.h>
#elif __linux__
    #include <inttypes.h>
    #include <unistd.h>
    #define __int64 int64_t
    #define _close close
    #define _read read
    #define _lseek64 lseek64
    #define _O_RDONLY O_RDONLY
    #define _open open
    #define _lseeki64 lseek64
    #define _lseek lseek
    #define stricmp strcasecmp
#endif

#define ORT_OLD_VISON 13  //ort1.12.0 ֮ǰ�İ汾Ϊ�ɰ汾API

struct yolo_info {
	std::vector<std::string> classes_yolo = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
	"hair drier", "toothbrush" };
	std::vector<cv::Scalar> color;
};

struct PoseKeyPoint {
	float x = 0;
	float y = 0;
	float confidence = 0;
};

struct OutputParams {
	int id;             //������id
	float confidence;   //������Ŷ�
	cv::Rect box;       //���ο�
	cv::RotatedRect rotatedBox;  //obb������ο�
	cv::Mat boxMask;       //���ο���mask����ʡ�ڴ�ռ�ͼӿ��ٶ�
	std::vector<PoseKeyPoint> keyPoints; //pose key points

};
struct MaskParams {
	//int segChannels = 32;
	//int segWidth = 160;
	//int segHeight = 160;
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;
};

struct PoseParams {
	float kptThreshold = 0.5;
	int kptRadius = 5;
	bool isDrawKptLine = true; //If True, the function will draw lines connecting keypoint for human pose.Default is True.
	cv::Scalar personColor = cv::Scalar(0, 0, 255);
	std::vector<std::vector<int>>skeleton = {
		{16, 14} ,{14, 12},{17, 15},{15, 13},
		{12, 13},{6, 12},{7, 13},{6, 7},{6, 8},{7, 9},
		{8, 10},{9, 11},{2, 3},{1, 2},{1, 3},{2, 4},
		{3, 5},{4, 6},{5, 7}
	};
	std::vector<cv::Scalar> posePalette =
	{
	cv::Scalar(255, 128, 0) ,
	cv::Scalar(255, 153, 51),
	cv::Scalar(255, 178, 102),
	cv::Scalar(230, 230, 0),
	cv::Scalar(255, 153, 255),
	cv::Scalar(153, 204, 255),
	cv::Scalar(255, 102, 255),
	cv::Scalar(255, 51, 255),
	cv::Scalar(102, 178, 255),
	cv::Scalar(51, 153, 255),
	cv::Scalar(255, 153, 153),
	cv::Scalar(255, 102, 102),
	cv::Scalar(255, 51, 51),
	cv::Scalar(153, 255, 153),
	cv::Scalar(102, 255, 102),
	cv::Scalar(51, 255, 51),
	cv::Scalar(0, 255, 0),
	cv::Scalar(0, 0, 255),
	cv::Scalar(255, 0, 0),
	cv::Scalar(255, 255, 255),
	};
	std::vector<int> limbColor = { 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16 };
	std::vector<int> kptColor = { 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9 };
	std::map<unsigned int, std::string> kptBodyNames{
					{0,"Nose"},
		{1,	"left_eye"},		{2,	"right_eye"},
		{3,	"left_ear"},		{4,	"right_ear"},
		{5,	"left_shoulder"},	{6,	"right_shoulder"},
		{7,	"left_elbow"},		{8,	"right_elbow"},
		{9,	"left_wrist"},		{10,"right_wrist"},
		{11,"left_hip"},		{12,"right_hip"},
		{13,"left_knee"},		{14,"right_knee"},
		{15,"left_ankle"},		{16,"right_ankle"}
	};
};



bool CheckModelPath(std::string modelPath);
bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize);
void DrawPred(cv::Mat& img,
	std::vector<OutputParams> result,
	std::vector<std::string> classNames,
	std::vector<cv::Scalar> color,
	bool isVideo = false
);
void DrawPredPose(cv::Mat& img, std::vector<OutputParams> result, PoseParams& poseParams, bool isVideo = false);

void DrawRotatedBox(cv::Mat& srcImg, cv::RotatedRect box, cv::Scalar color, int thinkness);
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
	cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
	const cv::Size& newShape = cv::Size(640, 640),
	bool autoShape = false,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(114, 114, 114));
void GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, std::vector<OutputParams>& output, const MaskParams& maskParams);
void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputParams& output, const MaskParams& maskParams);
int BBox2Obb(float centerX, float centerY, float boxW, float boxH, float angle, cv::RotatedRect& rotatedRect);
cv::Mat get_mask(const cv::Mat& img, std::vector<OutputParams> result);
