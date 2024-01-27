#ifndef CFTRACKER_FEATURE_DATA_HPP
#define CFTRACKER_FEATURE_DATA_HPP

#include "cftracker/common/datatype.hpp"


namespace cftracker {

// feature[Num_features][Dimension_of_the_feature];
typedef std::vector<std::vector<cv::Mat>> FEAT_DATA;

} // namespace cftracker

#endif // CFTRACKER_FEATURE_DATA_HPP

