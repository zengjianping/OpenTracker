#ifndef CFTRACKER_METRICS_HPP
#define CFTRACKER_METRICS_HPP

#include "cftracker/common/common.hpp"


namespace cftracker {

class Metrics {
public:
    float center_error(const cv::Rect2f bbox, const cv::Rect2f bboxGroundtruth);
    float iou(const cv::Rect2f bbox, const cv::Rect2f bboxGroundtruth);
    cv::Rect2f intersection(const cv::Rect2f bbox, const cv::Rect2f bboxGroundtruth);
    float auc();
};

} // namespace cftracker

#endif // CFTRACKER_METRICS_HPP

