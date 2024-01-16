#ifndef FEATURE_DATA_HPP
#define FEATURE_DATA_HPP

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <numeric>
#include <opencv2/core/core.hpp>


namespace cftracker {

// feature[Num_features][Dimension_of_the_feature];
typedef std::vector<std::vector<cv::Mat>> FEAT_DATA;
typedef cv::Vec<float, 2> COMPLEX; // represent a complex number;

} // namespace cftracker

#endif // FEATURE_DATA_HPP

