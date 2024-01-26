
#include "base_tracker.hpp"
#include "eco/eco_tracker.hpp"


namespace cftracker {

std::shared_ptr<BaseTracker> BaseTracker::CreateInstance(Type tracker_type, 
        std::string config_file) {
    std::shared_ptr<BaseTracker> tracker(nullptr);
    if (tracker_type == ECO) {
        tracker = std::make_shared<EcoTracker>(config_file);
    }
    return tracker;
}

BaseTracker::BaseTracker() {
}

BaseTracker::~BaseTracker() {
}

void BaseTracker::init(const cv::Mat& frame, const cv::Rect2f& roi) {
}

bool BaseTracker::update(const cv::Mat& frame, cv::Rect2f& roi) {
    return false;
}

} // namespace cftracker

