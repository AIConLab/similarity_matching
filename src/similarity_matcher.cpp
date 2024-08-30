#include "similarity_matcher.hpp"

SimilarityMatcher::SimilarityMatcher(ros::NodeHandle &nh) : _nh(nh), _stop_thread(false)
{
    if (pipe(_pipe_to_python) == -1 || pipe(_pipe_from_python) == -1) {
        ROS_ERROR("Failed to create pipes");
        return;
    }

    pid_t pid = fork();
    if (pid == 0) {  // Child process
        close(_pipe_to_python[1]);
        close(_pipe_from_python[0]);
        dup2(_pipe_to_python[0], STDIN_FILENO);
        dup2(_pipe_from_python[1], STDOUT_FILENO);
        
        std::string package_path = ros::package::getPath("similarity_matching");
        std::string cmd = package_path + "/libs/image_comparison_package/similarity_scorer_realtime.py";

        ROS_INFO("Executing Python script: %s", cmd.c_str());
        execlp("python3", "python3", cmd.c_str(), NULL);

        exit(1);
    } else if (pid > 0) {  // Parent process
        ROS_INFO("Python process started with PID: %d", pid);
        close(_pipe_to_python[0]);
        close(_pipe_from_python[1]);
    } else {
        ROS_ERROR("Fork failed");
        return;
    }

    _camera_stream_sub = _nh.subscribe("camera_image_stream", 1, &SimilarityMatcher::_camera_stream_callback, this);
    _target_image_sub = _nh.subscribe("target_image_updates", 1, &SimilarityMatcher::_target_image_callback, this);

    _current_target_image_pub = _nh.advertise<sensor_msgs::Image>("current_target_image", 1);
    _current_scene_image_pub = _nh.advertise<sensor_msgs::Image>("current_scene_image", 1);
    _current_similarity_score_pub = _nh.advertise<std_msgs::Float32>("current_similarity_score", 1);
    _current_target_keypoints_pub = _nh.advertise<similarity_matching::Keypoints>("current_target_keypoints", 1);
    _current_scene_keypoints_pub = _nh.advertise<similarity_matching::Keypoints>("current_scene_keypoints", 1);

    _processing_thread = std::thread(&SimilarityMatcher::_process_images, this);
}

SimilarityMatcher::~SimilarityMatcher()
{
    _stop_thread = true;
    if (_processing_thread.joinable()) {
        _processing_thread.join();
    }
    close(_pipe_to_python[1]);
    close(_pipe_from_python[0]);
}


bool SimilarityMatcher::_are_images_ready()
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    return !_target_image.empty() && !_latest_scene_image.empty();
}

void SimilarityMatcher::_target_image_callback(const sensor_msgs::Image::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    try {
        _target_image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } 
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception in target image callback: %s", e.what());
    }
}


void SimilarityMatcher::_camera_stream_callback(const sensor_msgs::Image::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    _latest_scene_image = cv_bridge::toCvCopy(msg, "bgr8")->image;
}

bool SimilarityMatcher::_write_image_to_python(const cv::Mat& image)
{
    int32_t header[3] = {image.rows, image.cols, image.channels()};
    ssize_t total_written = 0;
    while (static_cast<size_t>(total_written) < sizeof(header)) {
        ssize_t written = write(_pipe_to_python[1], reinterpret_cast<char*>(header) + total_written, sizeof(header) - total_written);
        if (written == -1) {
            if (errno == EINTR) continue;
            ROS_ERROR("Failed to write header to pipe: %s", strerror(errno));
            return false;
        }
        total_written += written;
    }

    size_t image_size = image.total() * image.elemSize();
    total_written = 0;
    while (static_cast<size_t>(total_written) < image_size) {
        ssize_t written = write(_pipe_to_python[1], image.data + total_written, image_size - total_written);
        if (written == -1) {
            if (errno == EINTR) continue;
            ROS_ERROR("Failed to write image data to pipe: %s", strerror(errno));
            return false;
        }
        total_written += written;
    }

    return true;
}

bool SimilarityMatcher::_read_python_output()
{
    float similarity_score;
    std::vector<cv::KeyPoint> target_keypoints, scene_keypoints;

    if (read(_pipe_from_python[0], &similarity_score, sizeof(float)) != sizeof(float)) {
        ROS_ERROR("Failed to read similarity score from Python");
        return false;
    }

    uint32_t target_kp_count, scene_kp_count;
    if (read(_pipe_from_python[0], &target_kp_count, sizeof(uint32_t)) != sizeof(uint32_t)) {
        ROS_ERROR("Failed to read target keypoint count from Python");
        return false;
    }

    for (uint32_t i = 0; i < target_kp_count; ++i) {
        float x, y;
        if (read(_pipe_from_python[0], &x, sizeof(float)) != sizeof(float) ||
            read(_pipe_from_python[0], &y, sizeof(float)) != sizeof(float)) {
            ROS_ERROR("Failed to read target keypoint coordinates from Python");
            return false;
        }
        target_keypoints.emplace_back(x, y, 1);
    }

    if (read(_pipe_from_python[0], &scene_kp_count, sizeof(uint32_t)) != sizeof(uint32_t)) {
        ROS_ERROR("Failed to read scene keypoint count from Python");
        return false;
    }

    for (uint32_t i = 0; i < scene_kp_count; ++i) {
        float x, y;
        if (read(_pipe_from_python[0], &x, sizeof(float)) != sizeof(float) ||
            read(_pipe_from_python[0], &y, sizeof(float)) != sizeof(float)) {
            ROS_ERROR("Failed to read scene keypoint coordinates from Python");
            return false;
        }
        scene_keypoints.emplace_back(x, y, 1);
    }

    std::lock_guard<std::mutex> lock(_data_mutex);
    _latest_similarity_score = similarity_score;
    _target_keypoints = std::move(target_keypoints);
    _scene_keypoints = std::move(scene_keypoints);
    _all_pub_data_ready = true;

    return true;
}

void SimilarityMatcher::_process_images()
{
    while (!_stop_thread)
    {
        if (!_are_images_ready()) {
            ROS_INFO_THROTTLE(5, "Waiting for images... Target ready: %s, Scene ready: %s",
                              _target_image.empty() ? "No" : "Yes",
                              _latest_scene_image.empty() ? "No" : "Yes");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        cv::Mat target, scene;
        {
            std::lock_guard<std::mutex> lock(_data_mutex);
            target = _target_image.clone();
            scene = _latest_scene_image.clone();
        }

        bool success = _write_image_to_python(target) && _write_image_to_python(scene);

        if (success) {
            if (_read_python_output()) {
                _publish_data();
            } else {
                ROS_WARN("Failed to read output from Python");
            }
        } else {
            ROS_WARN("Failed to write images to Python");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void SimilarityMatcher::_publish_data()
{
    std::lock_guard<std::mutex> lock(_data_mutex);
    if (_all_pub_data_ready) {
        sensor_msgs::ImagePtr target_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", _target_image).toImageMsg();
        sensor_msgs::ImagePtr scene_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", _latest_scene_image).toImageMsg();
        
        std_msgs::Float32 score_msg;
        score_msg.data = _latest_similarity_score;
        
        similarity_matching::Keypoints target_kp_msg;
        similarity_matching::Keypoints scene_kp_msg;

        for (const auto& kp : _target_keypoints) {
            geometry_msgs::Point32 point;
            point.x = kp.pt.x;
            point.y = kp.pt.y;
            target_kp_msg.keypoints.push_back(point);
        }

        for (const auto& kp : _scene_keypoints) {
            geometry_msgs::Point32 point;
            point.x = kp.pt.x;
            point.y = kp.pt.y;
            scene_kp_msg.keypoints.push_back(point);
        }
        
        _current_target_image_pub.publish(target_msg);
        _current_scene_image_pub.publish(scene_msg);
        _current_similarity_score_pub.publish(score_msg);
        _current_target_keypoints_pub.publish(target_kp_msg);
        _current_scene_keypoints_pub.publish(scene_kp_msg);

        
        _all_pub_data_ready = false;
    }
}