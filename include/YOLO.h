#ifndef YOLO_H
#define YOLO_H

#include <iostream>
#include<opencv2/opencv.hpp>

#include<math.h>
#include "yolov8.h"
#include "yolov8_seg.h"
#include "yolov8_utils.h"
#include<time.h>


namespace ORB_SLAM3
{
class YOLO
{
public:
    YOLO(std::string model_path);

    ~YOLO(){}

    std::vector<OutputParams> detect(cv::Mat& img);
    cv::Mat remove_masks(cv::Mat& img, std::vector<OutputParams> result);

private:
    std::string model_path;
    cv::dnn::Net net;
    Yolov8Seg task;
};

} //namespace ORB_SLAM

#endif // YOLO_H
