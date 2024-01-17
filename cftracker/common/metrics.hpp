#ifndef METRICS_HPP
#define METRICS_HPP

#include <math.h>
#include <opencv2/core.hpp>


namespace cftracker {

class Metrics {
public:
    float center_error(const cv::Rect2f bbox, const cv::Rect2f bboxGroundtruth);
    float iou(const cv::Rect2f bbox, const cv::Rect2f bboxGroundtruth);
    cv::Rect2f intersection(const cv::Rect2f bbox, const cv::Rect2f bboxGroundtruth);
    float auc();
};

} // namespace cftracker

#endif // METRICS_HPP

