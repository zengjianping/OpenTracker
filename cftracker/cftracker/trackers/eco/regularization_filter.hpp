#ifndef CFTRACKER_REGULARIZATION_FILTER_HPP
#define CFTRACKER_REGULARIZATION_FILTER_HPP

#include "cftracker/common/datatype.hpp"
#include "cftracker/trackers/eco/eco_parameter.hpp"


namespace cftracker {

cv::Mat get_regularization_filter(cv::Size sz, cv::Size2f target_sz, const EcoParameters &params);

} // namespace cftracker

#endif // CFTRACKER_REGULARIZATION_FILTER_HPP
