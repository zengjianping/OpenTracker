
#include "cftracker/trackers/base_tracker.hpp"
#include "cftracker/trackers/eco/eco_tracker.hpp"
#include "cftracker/trackers/kcf/kcf_tracker.hpp"


namespace cftracker {

std::shared_ptr<BaseTracker> BaseTracker::CreateInstance(Type tracker_type, 
        std::string config_file) {
    std::shared_ptr<BaseTracker> tracker(nullptr);
    if (tracker_type == ECO) {
        tracker = std::make_shared<EcoTracker>(config_file);
    }
    else if (tracker_type == KCF) {
        tracker = std::make_shared<KcfTracker>(config_file, false);
    }
    else if (tracker_type == DSST) {
        tracker = std::make_shared<KcfTracker>(config_file, true);
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

void BaseTracker::release() {
}

} // namespace cftracker

