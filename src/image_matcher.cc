// image_matcher.cc
#include "image_matcher.hh"
#include <ros/package.h>
#include <stdexcept>

#include <unistd.h>

ImageMatcher::ImageMatcher() : python_process(nullptr) 
{
    std::string package_path = ros::package::getPath("similarity_mapping");
    std::string script_path = package_path + "/scripts/xfeat_interface.py";
    python_process = popen(("python3 " + script_path).c_str(), "w");
    if (!python_process) {
        throw std::runtime_error("Failed to start Python process");
    }
}

ImageMatcher::~ImageMatcher() 
{
    if (python_process) {
        pclose(python_process);
    }
}

uint8_t ImageMatcher::compare_images(const cv::Mat &target_image, const cv::Mat &scene_image) 
{
    // Ensure target image not empty
    if (target_image.empty()) {
        throw std::invalid_argument("Target image is empty");
    }

    // Ensure scene image not empty
    else if (scene_image.empty()) {
        throw std::invalid_argument("Scene image is empty");
    }

    else
    {
        // Write image dimensions
        fprintf(python_process, "%d %d %d %d %d %d\n", 
                target_image.rows, target_image.cols, target_image.channels(),
                scene_image.rows, scene_image.cols, scene_image.channels());


        // Write image data
        fwrite(target_image.data, 1, target_image.total() * target_image.elemSize(), python_process);
        fwrite(scene_image.data, 1, scene_image.total() * scene_image.elemSize(), python_process);
        fflush(python_process);
        
            // Read result
            float similarity;
            int retry_count = 0;
            while (retry_count < 5) {  // Try up to 5 times
                if (fscanf(python_process, "%f", &similarity) == 1) {
                    // Ensure similarity is in the range [0, 1]
                    similarity = std::max(0.0f, std::min(1.0f, similarity));
                    
                    // Convert to uint8_t (0-100 range)
                    return static_cast<uint8_t>(similarity * 100);
                }
                usleep(100000);  // Wait for 100ms before retrying
                retry_count++;
            }
            throw std::runtime_error("Failed to read similarity score from Python process after multiple attempts");
        }
}
