#pragma once
#include <opencv2/opencv.hpp>
#include <cstdint>

class ImageMatcher
{
public:
    ImageMatcher();
    ~ImageMatcher();
    uint8_t compare_images(const cv::Mat &target_image, const cv::Mat &scene_image);
private:
    FILE* python_process;
};
