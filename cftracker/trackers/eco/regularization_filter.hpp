#ifndef REGULARIZATION_FILTER_HPP
#define REGULARIZATION_FILTER_HPP

#include <opencv2/opencv.hpp>
#include <cmath>
#include "eco_parameter.hpp"


namespace cftracker {

cv::Mat get_regularization_filter(cv::Size sz, cv::Size2f target_sz, const EcoParameters &params);

} // namespace cftracker

#endif // REGULARIZATION_FILTER_HPP
