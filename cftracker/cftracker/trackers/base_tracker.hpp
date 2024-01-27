#ifndef CFTRACKER_BASE_TRACKER_HPP
#define CFTRACKER_BASE_TRACKER_HPP

#include "cftracker/common/common.hpp"


namespace cftracker {

class BaseTracker {
public:
    enum Type {
        ECO = 0,
        KCF = 1,
        DSST = 2
    };
    static std::shared_ptr<BaseTracker> CreateInstance(Type tracker_type, std::string config_file);

public:
    BaseTracker();
    virtual ~BaseTracker();

    virtual void init(const cv::Mat& frame, const cv::Rect2f& roi); 
    virtual bool update(const cv::Mat& frame, cv::Rect2f& roi);
};

} // namespace cftracker

#endif // CFTRACKER_BASE_TRACKER_HPP

