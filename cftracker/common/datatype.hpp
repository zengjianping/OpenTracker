#ifndef DATATYPE_HPP
#define DATATYPE_HPP

#include <vector>
#include <string>
#include <opencv2/core.hpp>


namespace cftracker {

#define INF 0x7f800000 //0x7fffffff

typedef cv::Vec<float, 2> COMPLEX; // represent a complex number

} // namespace cftracker

#endif // DATATYPE_HPP

