#ifndef CFTRACKER_SCALE_FILTER_HPP
#define CFTRACKER_SCALE_FILTER_HPP

#include "cftracker/filters/scale_parameter.hpp"


namespace cftracker {

class ScaleFilter {
public:
    ScaleFilter(){};
    virtual ~ScaleFilter(){};
    
public:
    void init(int &nScales, float &scale_step, const ScaleParameters &params);
    float scale_filter_track(const cv::Mat &im, const cv::Point2f &pos, const cv::Size2f &base_target_sz,
        const float &currentScaleFactor, const ScaleParameters &params);
    cv::Mat extract_scale_sample(const cv::Mat &im, const cv::Point2f &posf, const cv::Size2f &base_target_sz,
        std::vector<float> &scaleFactors, const cv::Size &scale_model_sz);
    
private:
    std::vector<float> scaleSizeFactors_;
    std::vector<float> interpScaleFactors_;
    cv::Mat yf_;
    std::vector<float> window_;
};

} // namespace cftracker

#endif // CFTRACKER_SCALE_FILTER_HPP

