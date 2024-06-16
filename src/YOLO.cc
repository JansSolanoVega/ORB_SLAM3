#include "YOLO.h"
namespace ORB_SLAM3
{
    YOLO::YOLO(std::string _model_path):
                model_path(_model_path)
    {
        if (task.ReadModel(net, model_path, true)) {
            std::cout << "read net ok!" << std::endl;
        }
        else {
            std::cout << "read wrong!" << std::endl;
        }
    }

    std::vector<OutputParams> YOLO::detect(cv::Mat& img)
    {        
        std::vector<OutputParams> result;

        task.Detect(img, net, result);
        return result;
            
    }
    cv::Mat YOLO::remove_masks(cv::Mat& img, std::vector<OutputParams> result)
    {
        yolo_info info_yolo;
        cv::Mat mask = img.clone();
        for (int i = 0; i < result.size(); i++) { 
            if(info_yolo.classes_yolo[result[i].id]=="car"){
                if (result[i].boxMask.rows && result[i].boxMask.cols > 0)
                    mask(result[i].box).setTo(0, result[i].boxMask);
            }  
            
        }
        return mask;
            
    }

} //namespace ORB_SLAM